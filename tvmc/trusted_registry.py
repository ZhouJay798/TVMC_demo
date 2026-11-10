"""Locally simulated signed registration and witness checkpoints.

Keys must be pinned by the caller through an independent trusted channel.  This
module does not implement durable append-only storage, an independently operated
witness, trusted time, real-world identity, or a distributed consensus system.
"""
import copy
import math
import secrets
import time
from cryptography.exceptions import InvalidSignature
from .crypto import canonical, digest
from .bundle_delivery import outer_context
from .personal_delivery import context_for, replica_seed, verify_recipient, sha

REGISTRY_DOMAIN = 'TVMC/registry-snapshot/v1'
CHECKPOINT_DOMAIN = 'TVMC/registry-checkpoint/v1'
COMMIT_DOMAIN = 'TVMC/replica-code-commit/v1'


def code_commitment(replica_id, word, opening):
    bits = [int(bit) for bit in word]
    if len(bits) != 256 or any(bit not in (0, 1) for bit in bits):
        raise ValueError('a 256-bit codeword is required')
    if not isinstance(opening, str) or len(opening) != 64:
        raise ValueError('a 32-byte hexadecimal opening is required')
    try:
        bytes.fromhex(opening)
    except ValueError as exc:
        raise ValueError('invalid opening') from exc
    return digest([COMMIT_DOMAIN, replica_id, bits, opening])


def _sign(body, signer):
    return {'body': copy.deepcopy(body), 'signature': signer.sign(canonical(body)).hex()}


def verify_snapshot(snapshot, registry_public):
    try:
        body = snapshot['body']
        registry_public.verify(bytes.fromhex(snapshot['signature']), canonical(body))
        if (body['domain'] != REGISTRY_DOMAIN or body['version'] != 1 or
            not isinstance(body['registry_id'], str) or not body['registry_id'] or
            type(body['sequence']) is not int or body['sequence'] < 1 or
            body['stage'] not in ('precommit', 'issued') or
            not isinstance(body['entries'], list) or not body['entries']):
            raise ValueError('invalid registry snapshot')
        scheme=body['fingerprint_scheme']
        if (scheme['algorithm'] != 'Tardos' or type(scheme['candidates']) is not int or
            scheme['candidates'] != len(body['entries']) or scheme['length'] != 256 or
            not isinstance(scheme['p'], list) or len(scheme['p']) != 256 or
            not all(type(p) in (int, float) and math.isfinite(p) and 0 < p < 1 for p in scheme['p']) or
            type(scheme['threshold']) not in (int, float) or
            not math.isfinite(scheme['threshold']) or scheme['threshold'] <= 0):
            raise ValueError('invalid signed fingerprint parameters')
        if (len({entry['replica_id'] for entry in body['entries']}) != len(body['entries']) or
            len({entry['recipient'] for entry in body['entries']}) != len(body['entries'])):
            raise ValueError('duplicate registered replica or recipient')
        if body['stage'] == 'issued':
            _require_hash(body['manifest_hash'])
        return body
    except (KeyError, TypeError, InvalidSignature) as exc:
        raise ValueError('invalid registry signature or snapshot') from exc


def create_precommit(gate, admission, recipients, words, signer, tardos):
    gate.require_admitted(admission)
    if len(recipients) != len(admission.participants) or len(words) != len(recipients):
        raise ValueError('one recipient and codeword per participant required')
    if len({tuple(map(int, word)) for word in words}) != len(words):
        raise ValueError('duplicate codewords')
    biases = [float(value) for value in tardos.p]
    threshold = float(tardos.threshold)
    if (tardos.n != len(recipients) or tardos.length != 256 or len(biases) != tardos.length or
        not all(math.isfinite(value) and 0 < value < 1 for value in biases) or
        not math.isfinite(threshold) or threshold <= 0):
        raise ValueError('invalid fingerprint decoding parameters')
    entries, openings = [], {}
    for record, word in zip(recipients, words):
        verify_recipient(gate, admission, record)
        i = record['recipient']
        rid = f'{admission.sid}-U{i}'
        seed = replica_seed(admission, record)
        if [int(bit) for bit in word] != tardos.encode(seed).tolist():
            raise ValueError('codeword does not match its registered replica seed')
        opening = secrets.token_hex(32)
        openings[rid] = opening
        entries.append({'replica_id': rid, 'recipient': i,
                        'identity_commitment': dict(admission.identity_commitments)[i],
                        'recipient_key_sha256': sha(bytes.fromhex(record['public_key_der'])),
                        'fingerprint_seed': seed,
                        'code_commitment': code_commitment(rid, word, opening)})
    if {e['recipient'] for e in entries} != set(admission.participants):
        raise ValueError('recipient set differs from admitted participants')
    body = {'domain': REGISTRY_DOMAIN, 'version': 1, 'registry_id': 'local-demo',
            'sequence': 1, 'previous_hash': None, 'stage': 'precommit',
            'session': outer_context(admission), 'entries': entries,
            'fingerprint_scheme': {'algorithm': 'Tardos', 'candidates': tardos.n,
                                   'length': tardos.length, 'p': biases, 'threshold': threshold}}
    return _sign(body, signer), openings


def _manifest_entries(manifest, signature, issuer_public, gate, admission, recipients):
    issuer_public.verify(signature, canonical(manifest))
    gate.require_admitted(admission)
    if manifest.get('version') != 1 or manifest.get('session') != outer_context(admission):
        raise ValueError('issuer manifest session mismatch')
    by_id = {r['recipient']: r for r in recipients}
    entries = manifest.get('entries')
    if (set(by_id) != set(admission.participants) or len(by_id) != len(recipients) or
        not isinstance(entries, list) or len(entries) != len(by_id)):
        raise ValueError('issuer manifest recipient set mismatch')
    found = {}
    for index, item in enumerate(entries):
        ctx = item['context']; i = ctx['recipient']
        if i in found or i not in by_id:
            raise ValueError('duplicate or unknown recipient')
        verify_recipient(gate, admission, by_id[i])
        if ctx != context_for(admission, by_id[i], ctx['payload_sha256']):
            raise ValueError('issuer context mismatch')
        if item['cipher_path'] != f'{index}.cipher' or item['wrapped_path'] != f'{index}.key':
            raise ValueError('issuer manifest path mismatch')
        for field in ('payload_sha256',):
            _require_hash(ctx[field])
        for field in ('cipher_sha256', 'wrapped_sha256'):
            _require_hash(item[field])
        found[i] = item
    return found


def _require_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid SHA-256 digest')


def create_issuance(precommit, registry_public, manifest, issuer_signature, issuer_public,
                    gate, admission, recipients, signer):
    previous = verify_snapshot(precommit, registry_public)
    if previous['sequence'] != 1 or previous['previous_hash'] is not None or previous['stage'] != 'precommit':
        raise ValueError('invalid precommit history')
    if previous['session'] != outer_context(admission):
        raise ValueError('precommit session mismatch')
    try:
        items = _manifest_entries(manifest, issuer_signature, issuer_public, gate, admission, recipients)
        if (len(previous['entries']) != len(items) or
            {entry['recipient'] for entry in previous['entries']} != set(items)):
            raise ValueError('precommit recipient set mismatch')
        entries = []
        for pre in previous['entries']:
            item = items[pre['recipient']]; ctx = item['context']
            for key in ('replica_id', 'recipient', 'identity_commitment', 'recipient_key_sha256', 'fingerprint_seed'):
                if pre[key] != ctx[key]:
                    raise ValueError('issuer manifest disagrees with precommit')
            _require_hash(pre['code_commitment'])
            entries.append({**pre, 'payload_sha256': ctx['payload_sha256'],
                            'cipher_sha256': item['cipher_sha256'],
                            'wrapped_sha256': item['wrapped_sha256'], 'status': 'active'})
        body = {'domain': REGISTRY_DOMAIN, 'version': 1, 'registry_id': previous['registry_id'],
                'sequence': 2, 'previous_hash': digest(previous), 'stage': 'issued',
                'session': previous['session'], 'manifest_hash': digest(manifest), 'entries': entries,
                'fingerprint_scheme': previous['fingerprint_scheme']}
        return _sign(body, signer)
    except (KeyError, TypeError, InvalidSignature) as exc:
        raise ValueError('invalid issuer manifest') from exc


def verify_issuance(precommit, issuance, registry_public, manifest, issuer_signature, issuer_public,
                    gate, admission, recipients):
    # Reconstruct the exact permitted body without trusting metadata supplied by an untrusted verifier.
    pre = verify_snapshot(precommit, registry_public)
    issued = verify_snapshot(issuance, registry_public)
    if (pre['stage'] != 'precommit' or pre['sequence'] != 1 or pre['previous_hash'] is not None or
        issued['stage'] != 'issued' or issued['sequence'] != 2 or
        issued['previous_hash'] != digest(pre) or issued['registry_id'] != pre['registry_id'] or
        issued['session'] != pre['session'] or pre['session'] != outer_context(admission) or
        issued['fingerprint_scheme'] != pre['fingerprint_scheme'] or
        issued['manifest_hash'] != digest(manifest)):
        raise ValueError('registry history or manifest mismatch')
    try:
        items = _manifest_entries(manifest, issuer_signature, issuer_public, gate, admission, recipients)
        if (len(pre['entries']) != len(items) or len(issued['entries']) != len(items) or
            {entry['recipient'] for entry in pre['entries']} != set(items)):
            raise ValueError('registry recipient set mismatch')
        expected = []
        for item in pre['entries']:
            entry = items[item['recipient']]; ctx = entry['context']
            for key in ('replica_id', 'recipient', 'identity_commitment', 'recipient_key_sha256', 'fingerprint_seed'):
                if item[key] != ctx[key]:
                    raise ValueError('precommit and issuer binding mismatch')
            _require_hash(item['code_commitment'])
            expected.append({**item, 'payload_sha256': ctx['payload_sha256'],
                             'cipher_sha256': entry['cipher_sha256'],
                             'wrapped_sha256': entry['wrapped_sha256'], 'status': 'active'})
        if issued['entries'] != expected:
            raise ValueError('issued registry entries differ from issuer manifest')
        return issued
    except (KeyError, TypeError, InvalidSignature) as exc:
        raise ValueError('invalid signed issuance') from exc


def create_precommit_checkpoint(precommit, registry_public, witness_signer, challenge, now=None, ttl=300):
    pre = verify_snapshot(precommit, registry_public)
    if pre['stage'] != 'precommit' or pre['sequence'] != 1 or pre['previous_hash'] is not None:
        raise ValueError('invalid precommit history')
    return _checkpoint(pre, witness_signer, challenge, now, ttl)


def _checkpoint(snapshot_body, witness_signer, challenge, now, ttl):
    if (not isinstance(challenge, str) or len(challenge) != 64 or
        any(c not in '0123456789abcdef' for c in challenge) or
        type(ttl) is not int or not 1 <= ttl <= 300):
        raise ValueError('invalid checkpoint challenge or lifetime')
    now = int(time.time()) if now is None else now
    if type(now) is not int:
        raise ValueError('invalid clock')
    body = {'domain': CHECKPOINT_DOMAIN, 'registry_id': snapshot_body['registry_id'],
            'sequence': snapshot_body['sequence'], 'snapshot_hash': digest(snapshot_body),
            'challenge': challenge, 'issued_at': now, 'expires_at': now + ttl}
    return _sign(body, witness_signer)


def create_checkpoint(precommit, issuance, registry_public, witness_signer, challenge,
                      precommit_checkpoint, precommit_challenge, now=None, ttl=300):
    pre = verify_snapshot(precommit, registry_public)
    issued = verify_snapshot(issuance, registry_public)
    if (pre['sequence'] != 1 or pre['stage'] != 'precommit' or pre['previous_hash'] is not None or
        issued['sequence'] != 2 or issued['stage'] != 'issued' or issued['previous_hash'] != digest(pre) or
        issued['registry_id'] != pre['registry_id'] or issued['session'] != pre['session']):
        raise ValueError('snapshot is not a valid append')
    # The historical precommit checkpoint is verified at its own issuance time;
    # it anchors the prior snapshot, not the freshness of the current one. This
    # is an in-process witness simulation, not an independent trusted timestamp.
    try:
        prior_time = precommit_checkpoint['body']['issued_at']
    except (KeyError, TypeError) as exc:
        raise ValueError('missing precommit checkpoint') from exc
    verify_checkpoint(precommit_checkpoint, witness_signer.public_key(), precommit,
                      registry_public, precommit_challenge, now=prior_time)
    return _checkpoint(issued, witness_signer, challenge, now, ttl)


def verify_checkpoint(checkpoint, witness_public, issuance, registry_public, expected_challenge, now=None):
    try:
        issued = verify_snapshot(issuance, registry_public)
        body = checkpoint['body']
        witness_public.verify(bytes.fromhex(checkpoint['signature']), canonical(body))
        now = int(time.time()) if now is None else now
        if (type(now) is not int or type(body['issued_at']) is not int or type(body['expires_at']) is not int or
            body['domain'] != CHECKPOINT_DOMAIN or body['registry_id'] != issued['registry_id'] or
            body['sequence'] != issued['sequence'] or body['snapshot_hash'] != digest(issued) or
            body['challenge'] != expected_challenge or not isinstance(expected_challenge, str) or
            len(expected_challenge) != 64 or any(c not in '0123456789abcdef' for c in expected_challenge) or
            not body['issued_at'] <= now < body['expires_at'] or
            not 1 <= body['expires_at'] - body['issued_at'] <= 300):
            raise ValueError('checkpoint does not confirm this snapshot, challenge or time')
        return True
    except (KeyError, TypeError, InvalidSignature) as exc:
        raise ValueError('invalid checkpoint') from exc


def verify_recipient_binding(issued_body, replica_id, recipient, recipient_public_der):
    if issued_body.get('stage') != 'issued':
        raise ValueError('only an issued snapshot can authorize a recipient')
    matches = [e for e in issued_body['entries'] if e['replica_id'] == replica_id]
    if (len(matches) != 1 or matches[0]['recipient'] != recipient or matches[0]['status'] != 'active' or
        matches[0]['recipient_key_sha256'] != sha(recipient_public_der)):
        raise ValueError('replica does not belong to this recipient and public key')
    return matches[0]
