"""Independently recheck saved multiprocess records using supplied trust anchors.

Keys must be authenticated by the auditor through a separate trusted channel.
No checkpoint or independent witness is provided by this demo: currentness,
rollback resistance and split-view resistance are not established. Score
outputs were not retained, so this audit cannot re-run fingerprint extraction,
authenticate a model endpoint, or establish a real person's identity.
"""
import argparse
import hashlib
import json
import os
import stat
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from tvmc.crypto import P, Q, canonical
from tvmc.epochs import EpochPublic
from tvmc.identity_gate import IdentityGate
from tvmc.personal_delivery import replica_seed
from tvmc.reconstruction import request_context, verify_receipt
from tvmc.tardos import Tardos
from tvmc.trusted_registry import (code_commitment, verify_issuance,
                                   verify_recipient_binding, verify_snapshot)


def _open_member(root, relative):
    """Open a regular in-run file without following any directory/file symlinks.

    Only fixed, code-generated relative paths are accepted. Directory fds and
    O_NOFOLLOW avoid trusting path strings in metrics and reject symlink swaps.
    This does not protect against a same-UID actor modifying an already opened
    regular file or against replacement after the audit has finished.
    """
    path = PurePosixPath(relative)
    if (not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY') or
        path.is_absolute() or not path.parts or '..' in path.parts or '\\' in relative):
        raise ValueError('unsupported or unsafe audit path')
    with ExitStack() as stack:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, fd)
        for component in path.parts[:-1]:
            fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            stack.callback(os.close, fd)
        file_fd = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            os.close(file_fd)
            raise ValueError('audit input must be a regular file')
        return os.fdopen(file_fd, 'rb')


def _json(root, relative):
    with _open_member(root, relative) as stream:
        if os.fstat(stream.fileno()).st_size > 4 * 1024 * 1024:
            raise ValueError('audit metadata exceeds the maximum size')
        return json.load(stream)


def _sha(root, relative):
    digest = hashlib.sha256()
    with _open_member(root, relative) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _public_key(value):
    if not isinstance(value, str):
        raise ValueError('trust-anchor public keys must be hexadecimal strings')
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(value))


def audit_directory(directory, trusted_anchors):
    """Recheck records, ciphertext and recipient payloads against pinned keys.

    trusted_anchors is a caller-supplied JSON file with keys `issuer` and
    `registrar`. No file inside the run directory is automatically trusted.
    """
    root = os.path.abspath(os.fspath(directory))
    with Path(trusted_anchors).open(encoding='utf-8') as stream:
        anchors = json.load(stream)
    if not isinstance(anchors, dict) or not {'issuer', 'registrar'} <= set(anchors):
        raise ValueError('caller-supplied issuer and registrar anchors required')
    issuer, registrar = _public_key(anchors['issuer']), _public_key(anchors['registrar'])
    registration = _json(root, 'public_registration.json')
    signed = _json(root, 'signed_manifest.json')
    precommit = _json(root, 'registry_precommit.json')
    issuance = _json(root, 'registry_issued.json')
    codebook = _json(root, 'audit_codebook.json')
    receipt = _json(root, 'reconstruction_receipt.json')
    metrics = _json(root, 'metrics.json')
    if (registration['issuer_public'] != anchors['issuer'] or signed['issuer_public'] != anchors['issuer'] or
        registration['registrar_public'] != anchors['registrar'] or signed['registrar_public'] != anchors['registrar']):
        raise ValueError('recorded keys do not match supplied trust anchors')

    admission_data = registration['admission']
    gate = IdentityGate(registration['enrollment'])
    admission = gate.register(admission_data['sid'], admission_data['epoch'], admission_data['participants'],
                              admission_data['base_sha256'], admission_data['nonce'], admission_data['proofs'])
    if canonical(asdict(admission)) != canonical(admission_data):
        raise ValueError('admission fields differ from verified registration')
    if tuple(admission.participants) != (1, 2, 3):
        raise ValueError('this audit expects the registered three-party demo')
    recipients = registration['recipients']
    manifest = signed['manifest']
    signature = bytes.fromhex(signed['signature'])
    body = verify_issuance(precommit, issuance, registrar, manifest, signature, issuer,
                           gate, admission, recipients)
    pre_body = verify_snapshot(precommit, registrar)

    ids = registration['replica_ids']
    expected_ids = [f'{admission.sid}-U{i}' for i in admission.participants]
    if (ids != expected_ids or codebook['replica_ids'] != expected_ids or
        not isinstance(recipients, list) or len(recipients) != 3 or
        [record['recipient'] for record in recipients] != list(admission.participants)):
        raise ValueError('recipient or replica ordering differs from the registered session')
    words = codebook['words']
    if (not isinstance(words, list) or len(words) != 3 or
        any(not isinstance(word, list) or len(word) != 256 or
            any(type(bit) is not int or bit not in (0, 1) for bit in word) for word in words) or
        len({tuple(word) for word in words}) != 3):
        raise ValueError('invalid or duplicate fingerprint codewords')
    tardos = Tardos(n=3)
    scheme = pre_body['fingerprint_scheme']
    if (scheme['p'] != tardos.p.tolist() or scheme['threshold'] != tardos.threshold or
        codebook['p'] != scheme['p'] or codebook['threshold'] != scheme['threshold']):
        raise ValueError('fingerprint parameters do not match the signed registry')
    openings = codebook['openings']
    if not isinstance(openings, dict) or set(openings) != set(ids):
        raise ValueError('fingerprint opening set differs from the registration')
    manifest_items = {entry['context']['recipient']: entry for entry in manifest['entries']}
    payload_checks = []
    for i, rid, word, recipient in zip(admission.participants, ids, words, recipients):
        entry = verify_recipient_binding(body, rid, i, bytes.fromhex(recipient['public_key_der']))
        if tardos.encode(replica_seed(admission, recipient)).tolist() != word:
            raise ValueError('fingerprint code differs from its registered derivation')
        if code_commitment(rid, word, openings[rid]) != entry['code_commitment']:
            raise ValueError('fingerprint commitment mismatch')
        manifest_item = manifest_items[i]
        # Never trust a caller-controlled or saved metrics payload_path.
        payload = f'custodians/U{i}/received_model.zip'
        if _sha(root, payload) != manifest_item['context']['payload_sha256']:
            raise ValueError(f'U{i} delivered model payload hash mismatch')
        if _sha(root, f'encrypted_transit/U{i}.cipher') != manifest_item['cipher_sha256']:
            raise ValueError(f'U{i} inner ciphertext hash mismatch')
        if _sha(root, f'encrypted_transit/U{i}.key') != manifest_item['wrapped_sha256']:
            raise ValueError(f'U{i} wrapped-key hash mismatch')
        payload_checks.append({'recipient': i, 'replica_id': rid, 'payload_hash_valid': True,
                               'cipher_hash_valid': True, 'wrapped_key_hash_valid': True,
                               'fingerprint_commitment_valid': True})

    public = EpochPublic(**registration['key_public'])
    if (public.n != 5 or public.threshold != 3 or tuple(public.participants) != tuple(admission.participants) or
        list(public.identities) != [gate.registry[i] for i in range(1, 6)] or
        any(type(c) is not int or not 1 < c < P or pow(c, Q, P) != 1 for c in public.commitments)):
        raise ValueError('recovery public keys do not match registered identities and policy')
    key_context = {'domain': 'TVMC/admitted-key-recovery/v1', 'sid': admission.sid, 'epoch': admission.epoch,
                   'participants': list(admission.participants), 'threshold': 3, 'seed': admission.seed,
                   'base_sha256': admission.base_sha256, 'nonce': admission.nonce}
    if registration['key_context'] != key_context:
        raise ValueError('recovery key context differs from registered session')
    cipher_hash = _sha(root, 'outer_bundle.aesgcm')
    nonce = receipt['record']['context']['request_nonce']
    expected_context = request_context(public, key_context, cipher_hash, manifest, nonce)
    verify_receipt(receipt, public, expected_context, issuer)

    reported_status = metrics['status']
    reported_checks = metrics['checks']
    if (reported_status not in ('completed', 'completed_with_failed_checks') or
        not isinstance(reported_checks, dict) or not reported_checks or
        not all(isinstance(name, str) and type(value) is bool for name, value in reported_checks.items())):
        raise ValueError('invalid self-reported run status or checks')
    expected_status = 'completed' if all(reported_checks.values()) else 'completed_with_failed_checks'
    if reported_status != expected_status:
        raise ValueError('self-reported status contradicts its checks')
    if metrics['base_sha256'] != admission.base_sha256:
        raise ValueError('self-reported base hash differs from the signed session')

    return {
        'audit_status': 'verified_saved_records',
        'run_reported_status': reported_status,
        'run_reported_checks': reported_checks,
        'run_reported_status_authenticated': False,
        'enrollment_admission_and_recipient_proofs_valid': True,
        'issuer_and_registrar_signatures_and_bindings_valid': True,
        'fingerprint_commitments_and_parameters_valid': True,
        'reconstruction_receipt_and_outer_ciphertext_hash_valid': True,
        'outer_ciphertext_decrypted_or_contents_verified': False,
        'recipient_file_decryption_origin_verified': False,
        'fingerprint_actually_embedded_in_model_verified': False,
        'recipient_payloads': payload_checks,
        'checkpoint_freshness_and_rollback_protection': 'not_available',
        'request_nonce_freshness_and_replay_protection': 'not_verified',
        'trace_and_model_provenance': 'not_verified_no_raw_trace_or_remote_attestation',
        'warning': ('Keys must be authenticated independently; values stored with this local run do not '
                    'constitute independent trust anchors. All roles use the same OS user. The receipt '
                    'is the issuer-signed local service statement, not proof of every reconstruction. '
                    'The saved scores/status are self-reported; no raw trace responses were saved. '
                    'No session key remains to open the outer ciphertext, so its internal contents and '
                    'recipient files\' actual decryption origin were not independently verified. '
                    'Matching fingerprint commitments do not prove that model weights contain the code.')
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--trusted-anchors', required=True, type=Path,
                        help='independently authenticated JSON keys: issuer and registrar')
    args = parser.parse_args()
    try:
        result = audit_directory(args.run_dir, args.trusted_anchors)
    except (ValueError, KeyError, TypeError, OSError, InvalidSignature) as exc:
        parser.exit(2, f'audit failed: {type(exc).__name__}: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # An authenticated log structure does not turn self-reported failed checks
    # into a passed experiment. The JSON makes this distinction explicit.
    return 0 if result['run_reported_status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
