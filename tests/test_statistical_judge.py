import copy
from dataclasses import asdict
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
from tvmc.experiment import core_experiment
from tvmc.bridge import Collector,OneShotBox,evidence_for
from tvmc.crypto import prove,canonical,digest
from tvmc.inheritance import key_vector
from tvmc.statistical_judge import verify_statistical


def test_statistical_receipt_binding_and_signature():
    _,s=core_experiment(trials=1);rec=s['records'][0];c=Collector()
    receipt,sig=c.collect(OneShotBox(lambda _:s['words'][0]),[0]*256,'batch','endpoint','nonce')
    ev=evidence_for(rec,s['words'][0],s['openings'][0],receipt,sig)
    kc={'sid':rec.sid,'epoch':rec.epoch,'sha256':'test'}
    ctx={'public':asdict(s['public']),'records':[asdict(r) for r in s['records']],
         'collector_key':c.public_key.public_bytes(Encoding.Raw,PublicFormat.Raw).hex(),
         'endpoint':'endpoint','nonce':'nonce','now':s['now'],'tardos_p':s['tardos'].p.tolist(),
         'threshold':s['tardos'].threshold,'key_context':kc,
         'key_proofs':[prove(s['public'],s['shares'][i-1],kc) for i in rec.participants],
         'statistical_key_seeds':[60101,60202,60303,60404,60505],'statistical_vocab':1024}
    sr={'endpoint':'endpoint','nonce':'nonce','context_sha256':digest(ctx),'drift':key_vector(1024,60101).tolist()}
    evidence={'receipt':sr,'signature':c._key.sign(canonical(sr)).hex()}
    v=verify_statistical(ctx,ev,evidence)
    assert v['key_layer'] and v['statistical_deployment_layer'] and v['bridge_layer']
    edited=copy.deepcopy(evidence);edited['receipt']['drift'][0]+=1
    assert not verify_statistical(ctx,ev,edited)['bridge_layer']
    altered=copy.deepcopy(ctx);altered['statistical_key_seeds'][0]=9
    assert not verify_statistical(altered,ev,evidence)['bridge_layer']
    altered=copy.deepcopy(ctx);altered['nonce']='other'
    assert not verify_statistical(altered,ev,evidence)['bridge_layer']
