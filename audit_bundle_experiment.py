"""Public metadata/transcript consistency audit; outer ciphertext is not retained.
Pinned issuer key is supplied by a locally trusted registration snapshot; this
script cannot authenticate that snapshot's real-world provenance or HTTP server.
"""
import argparse,json
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from tvmc.crypto import canonical,verify_proof
from tvmc.epochs import EpochPublic
from tvmc.identity_gate import IdentityGate
from tvmc.personal_delivery import verify_recipient,replica_seed,context_for
from tvmc.bundle_delivery import outer_context
from tvmc.tardos import Tardos
from tvmc.reconstruction import request_context,verify_receipt
from tvmc.robustness import aggregate


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();out=a.run_dir
    r=json.loads((out/'registration_before_training.json').read_text());record=r['admission'];gate=IdentityGate(r['enrollment'])
    adm=gate.register(record['sid'],record['epoch'],record['participants'],record['base_sha256'],record['nonce'],r['request_proofs'])
    if adm.seed!=record['seed']:raise ValueError('admission mismatch')
    pub=EpochPublic(**r['key_public']);ctx=r['key_context'];proofs=r['key_proofs']
    if (pub.threshold!=len(adm.participants) or list(pub.participants)!=list(adm.participants) or
        list(pub.identities)!=[gate.registry[i] for i in range(1,6)] or ctx['sid']!=adm.sid or ctx['epoch']!=adm.epoch or
        ctx['participants']!=list(adm.participants) or ctx['threshold']!=pub.threshold or ctx['seed']!=adm.seed or
        ctx['nonce']!=adm.nonce or ctx['base_sha256']!=adm.base_sha256 or len(proofs)!=len(adm.participants) or
        {x['i'] for x in proofs}!=set(adm.participants) or not all(verify_proof(pub,x,ctx) for x in proofs)):
        raise ValueError('key proofs mismatch')
    t=Tardos(n=3)
    if not np.array_equal(t.p,r['p']) or t.threshold!=r['threshold']:raise ValueError('decoder changed')
    signed=json.loads((out/'signed_manifest.json').read_text());manifest=signed['manifest']
    if signed['issuer_public_key']!=r['issuer_public_key']:raise ValueError('issuer mismatch')
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(r['issuer_public_key'])).verify(bytes.fromhex(signed['signature_hex']),canonical(manifest))
    if manifest['session']!=outer_context(adm):raise ValueError('session mismatch')
    receipt_result=None
    receipt_path=out/'reconstruction_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text());metrics=json.loads((out/'model_metrics.json').read_text())
        expected=request_context(pub,ctx,metrics['outer_cipher_sha256'],manifest,receipt['record']['context']['request_nonce'])
        receipt_result=verify_receipt(receipt,pub,expected,Ed25519PublicKey.from_public_bytes(bytes.fromhex(r['issuer_public_key'])))
    results=[]
    for i,recipient in enumerate(r['recipients']):
        verify_recipient(gate,adm,recipient);seed=replica_seed(adm,recipient);sid=r['replica_ids'][i]
        if sid!=f'{adm.sid}-U{recipient["recipient"]}' or seed!=r['seeds'][i] or not np.array_equal(t.encode(seed),r['words'][i]):raise ValueError('fingerprint registration mismatch')
        item=manifest['entries'][i]
        if item['context']!=context_for(adm,recipient,item['context']['payload_sha256']):raise ValueError('manifest binding mismatch')
        trace=json.loads((out/(sid+'_batch_http.json')).read_text());values=trace['response']['score_differences']
        if len(values)!=768 or not np.isfinite(values).all():raise ValueError('invalid HTTP scores')
        bits=aggregate(values,3);decoded=[r['replica_ids'][j] for j in t.decode(r['words'],bits)]
        if decoded!=trace['result']['decoded_replicas']:raise ValueError('decoding mismatch')
        results.append({'replica':sid,'identity_key_binding_valid':True,'issuer_manifest_signature_valid':True,'decoded':decoded})
    print(json.dumps({'results':results,'reconstruction_receipt':receipt_result,'outer_ciphertext_rechecked':False,'scope':'trusted local snapshot and HTTP, full outer bytes not retained'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
