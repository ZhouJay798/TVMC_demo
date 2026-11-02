"""Five fixture controls: full shares can be interpolated outside a logging service."""
import argparse,json
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from run_personal_delivery import setup
from tvmc.bundle_delivery import make_inner,seal_bundle,open_bundle,open_inner
from tvmc.epochs import interpolate

p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args();rows=[]
for repeat in range(5):
    gate,adm,(key,pub,shares,proofs,ctx),keys,records,_=setup('a'*64)
    signer=Ed25519PrivateKey.generate();payload=b'offline fixture control, not a full model'
    inner=[make_inner(gate,adm,r,payload) for r in records]
    outer,_,_=seal_bundle(gate,adm,records,inner,key,signer)
    # No prove(), recover(), reconstruct(), or receipt service is called here.
    stolen_key=interpolate(shares)
    decoded,_=open_bundle(gate,adm,records,outer,stolen_key,signer.public_key())
    rows.append({'repeat':repeat+1,'offline_key_matches':stolen_key==key,
        'owner_private_key_still_required':True,
        'own_payload_decrypts':[open_inner(e,keys[i])==payload for i,e in enumerate(decoded)],
        'reconstruction_receipt_emitted':False})
(args.run_dir/'c1_offline_control.json').write_text(json.dumps({'scope':'five new independent small fixture trials, no new model training','trials':rows},indent=2)+'\n')
print(json.dumps(rows))
