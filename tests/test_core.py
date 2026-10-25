import copy
from dataclasses import replace
from itertools import combinations
import numpy as np
import pytest
from tvmc.crypto import (share_key, prove, recover, verify_proof, encrypt, decrypt, P, Q)
from tvmc.tardos import Tardos, collude
from tvmc.bridge import Registry, Collector, OneShotBox, evidence_for, judge


@pytest.fixture(scope='module')
def key_material():
    k,shares,public = share_key()
    context = {'sid':'test','epoch':1,'asset':'test-asset'}
    proofs = [prove(public,s,context) for s in shares]
    return k,shares,public,context,proofs


@pytest.mark.parametrize('a,b', list(combinations(range(5),2)))
def test_every_pair_recovers(key_material,a,b):
    k,s,p,c,proofs = key_material
    assert recover(p,[s[a],s[b]],[proofs[a],proofs[b]],c)==k


def test_reject_invalid_reconstruction(key_material):
    _,s,p,c,proofs = key_material
    for ss,pp in [(s[:1],proofs[:1]),([s[0],s[0]],[proofs[0],proofs[0]]),
                  ([replace(s[0],value=s[0].value+1),s[1]],proofs[:2]),
                  (s[:2],[dict(proofs[0],i=2),proofs[1]]), (s[:2],[])]:
        with pytest.raises(ValueError):
            recover(p,ss,pp,c)


def test_proof_binds_transcript(key_material):
    _,s,p,c,proofs = key_material
    assert P.bit_length()==2048 and pow(2,Q,P)==1
    assert verify_proof(p,proofs[0],c)
    assert not verify_proof(p,proofs[0],dict(c,epoch=2))
    for key in ('i','a','b','zf','zg'):
        altered = dict(proofs[0]); altered[key]+=1
        assert not verify_proof(p,altered,c)
    assert not verify_proof(p,dict(proofs[0],zf=Q),c)
    assert not verify_proof(p,{},c)


def test_aead_authentication(key_material):
    k,_,_,c,_ = key_material
    data = b'actual asset bytes'*200
    ct = encrypt(k,data,c)
    assert decrypt(k,ct,c)==data
    for bad_key,bad_ct,bad_aad in [(k+1,ct,c),(k,ct[:-1]+bytes([ct[-1]^1]),c),
                                  (k,ct,dict(c,epoch=2))]:
        with pytest.raises(Exception):
            decrypt(bad_key,bad_ct,bad_aad)


def test_score_formula_and_erasures():
    t = Tardos(length=256)
    p=t.p
    w=np.ones(256,dtype=np.int8)
    assert np.isclose(t.scores([w],w)[0],np.sqrt((1-p)/p).sum())
    assert np.isclose(t.scores([w],1-w)[0],-np.sqrt((1-p)/p).sum())
    assert t.decode([w],np.full(256,-1))==[]
    with pytest.raises(ValueError): t.decode([w],[0])


def test_identity_and_fragments(key_material):
    _,_,p,_,_ = key_material
    t,reg,collector=Tardos(),Registry(),Collector()
    rec,word,opening=reg.register('s',1,[1,2],p,t,100,1)
    receipt,sig=collector.collect(OneShotBox(lambda _:word),list(range(256)),'batch','endpoint','nonce')
    ev=evidence_for(rec,word,opening,receipt,sig)
    def check(e,registry=reg,now=1):
        return judge(registry,p,t,e,collector.public_key,'endpoint','nonce',now)
    assert check(ev)
    assert not check(dict(ev,seed='wrong'))
    assert not check(dict(ev,cm_w='wrong'))
    assert not check(ev,Registry())
    assert not check(ev,now=100)
    altered=copy.deepcopy(ev); altered['receipt']['observed'][0]^=1
    assert not check(altered)
    receipt,sig=collector.collect(OneShotBox(lambda _:word[:16]),list(range(16)),'batch','endpoint','nonce')
    assert not check(evidence_for(rec,word,opening,receipt,sig))
    with pytest.raises(ValueError): reg.register('s',1,[1,2],p,t,100,1)


@pytest.mark.parametrize('protocol,count',[('batch',256),('stream',256),('single',1)])
def test_one_shot_budget(protocol,count):
    box=OneShotBox(lambda prompts:np.array([p%2 for p in prompts]))
    assert len(box.query(list(range(256)),protocol))==count
    with pytest.raises(ValueError): box.query(list(range(256)),protocol)


def test_exported_public_verifier(key_material):
    from dataclasses import asdict
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from verify_evidence import verify
    _,_,public,key_context,proofs=key_material
    t,reg,collector=Tardos(),Registry(),Collector()
    rec,word,opening=reg.register('test',1,[1,2],public,t,100,1)
    receipt,sig=collector.collect(OneShotBox(lambda _:word),list(range(256)),'batch','endpoint','nonce')
    ev=evidence_for(rec,word,opening,receipt,sig)
    ctx={'public':asdict(public),'records':[asdict(rec)],'tardos_p':t.p.tolist(),
         'threshold':t.threshold,'collector_key':collector.public_key.public_bytes(Encoding.Raw,PublicFormat.Raw).hex(),
         'endpoint':'endpoint','nonce':'nonce','now':1,'key_proofs':proofs[:2],'key_context':key_context}
    assert all(verify(ctx,ev).values())
    assert not verify(ctx,dict(ev,cm_w='forged'))['deployment_layer']
    wrong=copy.deepcopy(ctx); wrong['key_context']['sid']='other'
    assert not verify(wrong,ev)['key_layer']
