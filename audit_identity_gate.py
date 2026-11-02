"""Independently verify consistency of public preregistration and admission transcripts.

The input snapshot itself must be independently trusted; no signature or real
identity authority is implied by a self-contained proof of secret knowledge.
"""
import argparse,hashlib,json
from pathlib import Path
from tvmc.crypto import verify_proof
from tvmc.epochs import EpochPublic
from tvmc.identity_gate import IdentityGate


def audit(data):
    results=[]
    for group in data['public_repeats']:
        gate=IdentityGate(group['enrollment'])
        for item in group['admissions']:
            recorded=item['admission'];ctx=item['request_context'];ids=recorded['participants']
            reconstructed=gate.register(recorded['sid'],recorded['epoch'],ids,recorded['base_sha256'],recorded['nonce'],item['request_proofs'])
            public=EpochPublic(**item['key_public']);keyctx=item['key_context'];proofs=item['key_proofs']
            ok=(reconstructed.seed==recorded['seed'] and list(map(list,reconstructed.identity_commitments))==recorded['identity_commitments'] and
                ctx==gate.context(recorded['sid'],recorded['epoch'],ids,recorded['base_sha256'],recorded['nonce']) and
                list(public.participants)==ids and public.threshold==keyctx.get('threshold')==len(ids) and public.n==5 and
                list(public.identities)==[gate.registry[i] for i in range(1,6)] and
                keyctx['sid']==recorded['sid'] and keyctx['epoch']==recorded['epoch'] and
                keyctx['participants']==ids and keyctx['seed']==recorded['seed'] and
                keyctx['base_sha256']==recorded['base_sha256'] and keyctx['nonce']==recorded['nonce'] and
                len(proofs)==len(ids) and {p['i'] for p in proofs}==set(ids) and
                all(verify_proof(public,p,keyctx) for p in proofs))
            results.append({'repeat':group['repeat'],'sid':recorded['sid'],'ok':bool(ok)})
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    data=json.loads((args.run_dir/'crypto_metrics.json').read_text());results=audit(data)
    print(json.dumps({'public_transcripts_checked':len(results),'passed':sum(r['ok'] for r in results),'results':results},indent=2))
    return 0 if all(r['ok'] for r in results) else 2

if __name__=='__main__':raise SystemExit(main())
