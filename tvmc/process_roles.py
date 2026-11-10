"""Local, separate-process cryptographic roles for a custody experiment.

Workers are fresh Python interpreters and keep RSA/Ed25519/identity private
keys in their own process. JSON over inherited stdin/stdout is a *trusted local*
control channel, not an authenticated network protocol or hardware isolation.
The dealer/coordinator sees session shares; a recovery coordinator receives
share values, but workers never export their identity secret or private keys.
The generic signing action is a research primitive, not a signing service.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding

from .bundle_delivery import make_inner, open_inner, seal_bundle
from .crypto import G, P, Q, Share, canonical, digest, prove, scalar
from .epochs import EpochPublic
from .identity_gate import Admission, IdentityGate, enrollment_context, identity_proof
from .personal_delivery import binding_context, new_key, public_bytes, replica_seed, sha
from .tardos import Tardos
from .trusted_registry import create_issuance, create_precommit, verify_issuance, verify_recipient_binding


class RoleError(RuntimeError):
    """A local worker rejected a request or exited."""


def encode_entry(entry):
    """Convert a small encrypted inner package into JSON-compatible values."""
    return {'context': entry['context'], 'ciphertext': entry['ciphertext'].hex(),
            'wrapped_key': entry['wrapped_key'].hex()}


def decode_entry(entry):
    if 'cipher_path' in entry:
        return {'context': entry['context'], 'ciphertext': Path(entry['cipher_path']).read_bytes(),
                'wrapped_key': Path(entry['wrapped_path']).read_bytes()}
    return {'context': entry['context'], 'ciphertext': bytes.fromhex(entry['ciphertext']),
            'wrapped_key': bytes.fromhex(entry['wrapped_key'])}


def _write_new(path, data):
    """Write a new owner-only file; do not overwrite a pre-existing target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
    except BaseException:
        # A partial file must never be mistaken for a completed artifact.
        path.unlink(missing_ok=True)
        raise
    return str(path.resolve())


def _ed_public(key):
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def _restore_gate(enrollment, admission):
    """Recheck public admission proofs inside each role's own process."""
    gate = IdentityGate(enrollment)
    admitted = gate.register(admission['sid'], admission['epoch'], admission['participants'],
                             admission['base_sha256'], admission['nonce'], admission['proofs'])
    if canonical(asdict(admitted)) != canonical(admission):
        raise ValueError('admission data differs from independently verified admission')
    return gate, admitted


class _Worker:
    def __init__(self, role, custodian_id):
        self.role = role
        self.id = custodian_id
        self.private = None
        self.gamma = None
        self.share = None
        self.public = None
        if role in ('issuer', 'admin'):
            self.private = ed25519.Ed25519PrivateKey.generate()
        elif role == 'custodian' and type(custodian_id) is int and 1 <= custodian_id <= 5:
            self.private = new_key()
            self.gamma = scalar()
        else:
            raise ValueError('invalid role or custodian id')

    def call(self, action, params):
        if action == 'hello':
            result = {'role': self.role, 'pid': os.getpid(), 'custodian_id': self.id}
            if self.role in ('issuer', 'admin'):
                result['public_key'] = _ed_public(self.private.public_key())
            else:
                result['public_key_der'] = public_bytes(self.private.public_key()).hex()
                result['identity_commitment'] = pow(G, self.gamma, P)
            return result
        if action == 'sign' and self.role in ('issuer', 'admin'):
            return {'signature': self.private.sign(bytes.fromhex(params['message'])).hex()}
        if action == 'build_payloads' and self.role == 'issuer':
            # The JSON-line stdout channel must not be polluted by model
            # libraries or training progress; diagnostics go to stderr.
            with redirect_stdout(sys.stderr):
                from .process_payloads import build_payloads
                return build_payloads(**params)
        if action in ('make_and_seal', 'make_and_seal_files') and self.role == 'issuer':
            gate, admission = _restore_gate(params['enrollment'], params['admission'])
            recipients = params['recipients']
            secret = params['session_secret']
            if type(secret) is not int or not 0 < secret < Q:
                raise ValueError('invalid session secret')
            if action == 'make_and_seal':
                values = params['payloads']
                if len(values) != len(recipients) or any(len(value) > 2 * 1024 * 1024 for value in values):
                    raise ValueError('JSON payload mode supports one payload up to 1 MiB per recipient')
                payloads = [bytes.fromhex(value) for value in values]
            else:
                paths = params['payload_paths']
                if len(paths) != len(recipients):
                    raise ValueError('one model package path per recipient required')
                payloads = [Path(path).read_bytes() for path in paths]
            entries = [make_inner(gate, admission, record, payload)
                       for record, payload in zip(recipients, payloads)]
            outer, manifest, signature = seal_bundle(gate, admission, recipients, entries, secret, self.private)
            result = {'manifest': manifest, 'signature': signature.hex(),
                      'outer_sha256': sha(outer), 'outer_bytes': len(outer)}
            if action == 'make_and_seal':
                result['outer'] = outer.hex()
            else:
                result['outer_path'] = _write_new(params['outer_path'], outer)
            return result
        if action == 'enrollment' and self.role == 'custodian':
            return {'i': self.id, 'commitment': pow(G, self.gamma, P),
                    'proof': identity_proof(self.id, self.gamma, enrollment_context(self.id))}
        if action == 'identity_proof' and self.role == 'custodian':
            return identity_proof(self.id, self.gamma, params['context'])
        if action == 'bind_recipient' and self.role == 'custodian':
            admission = Admission(**params['admission'])
            if (self.id not in admission.participants or
                dict(admission.identity_commitments).get(self.id) != pow(G, self.gamma, P)):
                raise ValueError('custodian identity does not match admission')
            der = public_bytes(self.private.public_key())
            ctx = binding_context(admission, self.id, pow(G, self.gamma, P), der)
            pop = self.private.sign(canonical(ctx), padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                          salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
            return {'recipient': self.id, 'public_key_der': der.hex(),
                    'identity_proof': identity_proof(self.id, self.gamma, ctx), 'key_possession': pop.hex()}
        if action == 'store_share' and self.role == 'custodian':
            public = EpochPublic(**params['public'])
            value = params['value']
            if type(value) is not int or not 0 <= value < Q:
                raise ValueError('invalid share value')
            share = Share(self.id, value, self.gamma)
            if not public.verify(share):
                raise ValueError('share or identity does not match public commitments')
            self.share, self.public = share, public
            return {'stored': True, 'i': self.id}
        if action == 'prove_share' and self.role == 'custodian':
            if self.share is None:
                raise ValueError('share not provisioned')
            proof = prove(self.public, self.share, params['context'])
            # The recovery coordinator receives the share value and proof but
            # never the identity secret gamma (normally embedded in Share).
            return {'i': self.id, 'value': self.share.value, 'proof': proof}
        if action == 'create_precommit' and self.role == 'admin':
            gate, admission = _restore_gate(params['enrollment'], params['admission'])
            recipients = params['recipients']
            tardos = Tardos(n=len(recipients))
            words = [tardos.encode(replica_seed(admission, record)) for record in recipients]
            precommit, openings = create_precommit(gate, admission, recipients, words, self.private, tardos)
            # These opening values are experiment-sensitive metadata; the
            # trusted parent must not publish them or treat them as private keys.
            return {'precommit': precommit, 'openings': openings}
        if action == 'create_issuance' and self.role == 'admin':
            gate, admission = _restore_gate(params['enrollment'], params['admission'])
            issuer = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(params['issuer_public']))
            return create_issuance(params['precommit'], self.private.public_key(), params['manifest'],
                    bytes.fromhex(params['issuer_signature']), issuer, gate, admission,
                    params['recipients'], self.private)
        if action == 'verified_open' and self.role == 'custodian':
            gate, admission = _restore_gate(params['enrollment'], params['admission'])
            issuer = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(params['issuer_public']))
            registry = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(params['registry_public']))
            sig = bytes.fromhex(params['issuer_signature'])
            manifest = params['manifest']
            body = verify_issuance(params['precommit'], params['issued'], registry, manifest, sig,
                                   issuer, gate, admission, params['recipients'])
            # Reject obvious cross-recipient deliveries before reading a large
            # ciphertext file. The file hashes are still checked after read.
            entry_context = params['entry']['context']
            rid = entry_context['replica_id']
            registered = verify_recipient_binding(body, rid, self.id, public_bytes(self.private.public_key()))
            matches = [item for item in manifest['entries'] if item['context']['replica_id'] == rid]
            if len(matches) != 1 or matches[0]['context'] != entry_context:
                raise ValueError('inner package context does not match the signed records')
            entry = decode_entry(params['entry'])
            if (sha(entry['ciphertext']) != matches[0]['cipher_sha256'] or
                sha(entry['wrapped_key']) != matches[0]['wrapped_sha256'] or
                registered['payload_sha256'] != entry['context']['payload_sha256'] or
                registered['cipher_sha256'] != matches[0]['cipher_sha256'] or
                registered['wrapped_sha256'] != matches[0]['wrapped_sha256']):
                raise ValueError('inner package does not match the signed records')
            plaintext = open_inner(entry, self.private)
            # Return only a digest/size and, optionally, an owner-only path.
            result = {'replica_id': rid, 'recipient': self.id, 'payload_sha256': sha(plaintext),
                      'payload_bytes': len(plaintext)}
            if params.get('output_path') is not None:
                result['payload_path'] = _write_new(params['output_path'], plaintext)
            return result
        raise ValueError('action is unavailable for this role')


def _worker_main(role, custodian_id):
    worker = _Worker(role, custodian_id)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request['action'] == 'close':
                print(json.dumps({'ok': True, 'result': {'closed': True}}), flush=True)
                break
            result = worker.call(request['action'], request.get('params', {}))
            print(json.dumps({'ok': True, 'result': result}, separators=(',', ':')), flush=True)
        except Exception as exc:
            print(json.dumps({'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)},
                             separators=(',', ':')), flush=True)


class RoleProcess:
    """Trusted parent-side proxy; responses intentionally exclude private keys."""

    def __init__(self, role, custodian_id=None):
        if role not in ('issuer', 'admin', 'custodian'):
            raise ValueError('invalid role')
        if role == 'custodian' and (type(custodian_id) is not int or not 1 <= custodian_id <= 5):
            raise ValueError('custodian id must be between 1 and 5')
        if role != 'custodian' and custodian_id is not None:
            raise ValueError('only custodian roles have an id')
        args = [sys.executable, '-u', '-m', 'tvmc.process_roles', '--worker', role]
        if custodian_id is not None:
            args += ['--custodian-id', str(custodian_id)]
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding='utf-8', bufsize=1)
        self.role = role
        self.info = self.call('hello')

    def call(self, action, **params):
        if self.process.poll() is not None:
            raise RoleError(f'{self.role} worker exited: {self.process.returncode}')
        self.process.stdin.write(json.dumps({'action': action, 'params': params}, separators=(',', ':'))+'\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RoleError(f'{self.role} worker ended without a response')
        response = json.loads(line)
        if not response['ok']:
            raise RoleError(f"{response['error_type']}: {response['error']}")
        return response['result']

    def sign(self, message):
        if self.role not in ('issuer', 'admin'):
            raise RoleError('only issuer/admin can sign using this proxy')
        return bytes.fromhex(self.call('sign', message=message.hex())['signature'])

    def public_key(self):
        if self.role not in ('issuer', 'admin'):
            raise RoleError('not an Ed25519 signer')
        return ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.info['public_key']))

    def close(self):
        if self.process.poll() is None:
            try:
                self.call('close')
                self.process.wait(timeout=5)
            except (OSError, ValueError, RoleError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('issuer', 'admin', 'custodian'), required=True)
    parser.add_argument('--custodian-id', type=int)
    args = parser.parse_args()
    _worker_main(args.worker, args.custodian_id)


if __name__ == '__main__':
    main()
