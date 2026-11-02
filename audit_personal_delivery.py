"""Public consistency audit against a caller-trusted local snapshot (not external authenticity)."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from tvmc.identity_gate import IdentityGate
from tvmc.epochs import EpochPublic
from tvmc.crypto import verify_proof
from tvmc.personal_delivery import verify_recipient,replica_seed,context_for
from tvmc.tardos import Tardos
from tvmc.robustness import aggregate


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args();out=args.run_dir
    r=json.loads((out/'registration_before_training.json').read_text());gate=IdentityGate(r['enrollment']);a=r['admission']
    adm=gate.register(a['sid'],a['epoch'],a['participants'],a['base_sha256'],a['nonce'],r['request_proofs'])
    if adm.seed!=a['seed']:raise ValueError('admission seed mismatch')
    pub=EpochPublic(**r['key_public']);ctx=r['key_context']
    if (pub.threshold!=len(adm.participants) or list(pub.participants)!=list(adm.participants) or
        list(pub.identities)!=[gate.registry[i] for i in range(1,6)] or
        ctx['sid']!=adm.sid or ctx['epoch']!=adm.epoch or ctx['participants']!=list(adm.participants) or
        ctx['threshold']!=pub.threshold or ctx['seed']!=adm.seed or ctx['nonce']!=adm.nonce or
        ctx['base_sha256']!=adm.base_sha256 or len(r['key_proofs'])!=len(adm.participants) or
        {x['i'] for x in r['key_proofs']}!=set(adm.participants) or
        not all(verify_proof(pub,x,ctx) for x in r['key_proofs'])):raise ValueError('key public proof mismatch')
    t=Tardos(n=3);records=r['recipients'];results=[]
    if len(records)!=3 or {x['recipient'] for x in records}!=set(adm.participants):raise ValueError('wrong recipients')
    if not np.array_equal(t.p,r['p']) or t.threshold!=r['threshold']:raise ValueError('changed decoder settings')
    for i,record in enumerate(records):
        verify_recipient(gate,adm,record);replica=r['replica_ids'][i]
        if replica!=f'{adm.sid}-U{record["recipient"]}' or replica_seed(adm,record)!=r['seeds'][i] or not np.array_equal(t.encode(r['seeds'][i]),r['words'][i]):raise ValueError('fingerprint binding mismatch')
        envelope=json.loads((out/(replica+'_envelope.json')).read_text())
        if envelope['context']!=context_for(adm,record,envelope['context']['payload_sha256']):raise ValueError('envelope binding mismatch')
        h=hashlib.sha256()
        with (out/(replica+'.aesgcm')).open('rb') as f:
            for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
        if h.hexdigest()!=envelope['cipher_sha256']:raise ValueError('cipher hash mismatch')
        transcript=json.loads((out/(replica+'_batch_http.json')).read_text());values=transcript['response']['score_differences']
        if len(values)!=768 or not np.isfinite(values).all():raise ValueError('invalid margins')
        bits=aggregate(values,3);decoded=[r['replica_ids'][j] for j in t.decode(r['words'],bits)]
        if decoded!=transcript['result']['decoded_replicas']:raise ValueError('decoder output mismatch')
        results.append({'replica':replica,'public_binding_valid':True,'cipher_hash_valid':True,'decoded_replicas':decoded})
    print(json.dumps({'scope':'public consistency under trusted local snapshot; HTTP authenticity not established','results':results},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
