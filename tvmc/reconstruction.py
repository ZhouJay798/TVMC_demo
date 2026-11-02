"""Nonce-bound reconstruction receipts. Requires an externally pinned service key.

A receipt attests service execution; knowledge proofs do not independently prove
execution or identify the human controlling a key. No secret shares are logged.
"""
from .crypto import canonical, digest, recover, verify_proof


def request_context(public, session_context, bundle_hash, manifest, nonce):
    if not isinstance(nonce, str) or len(nonce) < 32:
        raise ValueError('fresh request nonce required')
    return {'domain': 'TVMC/reconstruction/v1', 'session': session_context,
            'public_hash': digest(public.__dict__), 'bundle_hash': bundle_hash,
            'manifest_hash': digest(manifest), 'request_nonce': nonce}


def reconstruct(public, shares, proofs, context, signer):
    if (public.threshold != len(public.participants) or len(shares) != public.threshold or
        {s.i for s in shares} != set(public.participants) or
        context.get('domain') != 'TVMC/reconstruction/v1' or context.get('public_hash') != digest(public.__dict__)):
        raise ValueError('all registered participants required')
    key = recover(public, shares, proofs, context)
    record = {'context': context, 'participants': sorted(s.i for s in shares),
              'proofs': proofs, 'service_attestation': 'reconstruction_completed'}
    return key, {'record': record, 'signature': signer.sign(canonical(record)).hex()}


def verify_receipt(receipt, public, expected_context, service_public, seen_nonces=None):
    record = receipt['record']
    service_public.verify(bytes.fromhex(receipt['signature']), canonical(record))
    if (record['context'] != expected_context or
        expected_context['public_hash'] != digest(public.__dict__) or
        record['service_attestation'] != 'reconstruction_completed' or
        record['participants'] != sorted(public.participants) or
        len(record['proofs']) != public.threshold or
        {p['i'] for p in record['proofs']} != set(public.participants) or
        not all(verify_proof(public, p, expected_context) for p in record['proofs'])):
        raise ValueError('receipt binding or proof mismatch')
    nonce = expected_context['request_nonce']
    if seen_nonces is not None:
        if nonce in seen_nonces:
            raise ValueError('request already consumed')
        seen_nonces.add(nonce)
    return {'verified_participants': record['participants'],
            'scope': 'service-attested reconstruction and participant key knowledge',
            'individual_leaker_proven': False}
