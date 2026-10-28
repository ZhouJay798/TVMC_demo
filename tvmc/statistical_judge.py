"""Research statistical receipt verification, NOT the paper's Tardos guarantee.
Trusts an independently authenticated context and honest measurement collector.
"""
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
from .crypto import canonical, digest
from .inheritance import detect
from .bridge import identity_seed, commitment
from .crypto import Public
from verify_evidence import verify


def verify_statistical(context, natural_evidence, evidence):
    result={'key_layer':False,'statistical_deployment_layer':False,'bridge_layer':False,
            'scope':'research statistical attribution, not Tardos security'}
    try:
        result['key_layer']=verify(context,natural_evidence)['key_layer']
        receipt=evidence['receipt']
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(context['collector_key'])).verify(
            bytes.fromhex(evidence['signature']),canonical(receipt))
        if receipt['context_sha256']!=digest(context):return result
        if receipt['endpoint']!=context['endpoint'] or receipt['nonce']!=context['nonce']:return result
        rec=next(r for r in context['records'] if r['sid']==natural_evidence['sid'])
        if rec['expires']<=context['now']:return result
        if rec['seed']!=identity_seed(rec['sid'],rec['epoch'],rec['participants'],Public(**context['public'])):return result
        if natural_evidence['seed']!=rec['seed'] or commitment(natural_evidence['word'],natural_evidence['opening'])!=rec['cm_w']:return result
        drift=np.asarray(receipt['drift'],dtype=float)
        if drift.shape!=(context['statistical_vocab'],) or not np.isfinite(drift).all():return result
        detection=detect(drift,context['statistical_key_seeds'])
        result['decoded_users']=detection['decoded_users']
        result['statistical_deployment_layer']=detection['decoded_users']==[rec['consumer']]
        result['bridge_layer']=bool(result['key_layer'] and result['statistical_deployment_layer']
           and rec['sid']==context['key_context']['sid'] and rec['epoch']==context['key_context']['epoch']
           and set(rec['participants'])=={p['i'] for p in context['key_proofs']})
    except (ValueError,KeyError,TypeError,InvalidSignature,StopIteration):pass
    return result
