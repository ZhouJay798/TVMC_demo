"""Recovery from participant-submitted shares without sending identity secrets.

This is still a trusted in-memory recovery service: all share values and the
recovered secret are visible to the service. It is not threshold decryption,
distributed key generation, a durable replay store, or a production protocol.
"""
from types import SimpleNamespace

from .crypto import G, P, Q, canonical, digest, verify_proof
from .epochs import interpolate


def reconstruct_submissions(public, submissions, context, signer):
    """Verify full participant set, Feldman share commitments and proofs."""
    if (public.threshold != len(public.participants) or
        not isinstance(submissions, list) or len(submissions) != public.threshold or
        context.get('domain') != 'TVMC/reconstruction/v1' or
        context.get('public_hash') != digest(public.__dict__)):
        raise ValueError('all registered participants and correct context required')
    checked = []
    for item in submissions:
        if not isinstance(item, dict) or set(item) != {'i', 'value', 'proof'}:
            raise ValueError('invalid share submission')
        i, value, proof = item['i'], item['value'], item['proof']
        if (type(i) is not int or i not in public.participants or type(value) is not int or
            not 0 <= value < Q or not isinstance(proof, dict) or proof.get('i') != i or
            pow(G, value, P) != public.y(i) or not verify_proof(public, proof, context)):
            raise ValueError('invalid share value or identity knowledge proof')
        checked.append(SimpleNamespace(i=i, value=value))
    if {share.i for share in checked} != set(public.participants):
        raise ValueError('duplicate or missing registered participants')
    key = interpolate(checked)
    record = {'context': context, 'participants': sorted(public.participants),
              'proofs': [item['proof'] for item in submissions],
              'service_attestation': 'reconstruction_completed'}
    return key, {'record': record, 'signature': signer.sign(canonical(record)).hex()}
