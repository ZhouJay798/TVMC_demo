import copy
import json
import time
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey,X25519PublicKey
from tvmc import paper_hpke as hpke,paper_threshold as te
from tvmc.paper_workflow import Custody,sign,sha
from tvmc.paper_trace import escrow_parameters,disclose_parameters
from tvmc.stream_delivery import issue_file,receive_file,file_hash
from tvmc.transparency import Journal,root,leaf,inclusion,verify_inclusion,verify_extension,verify_checkpoint
from tvmc.evidence import collect_quorum,verify_quorum,ServedLog,verify_rebuttal
from tvmc.crypto import canonical


@pytest.fixture(scope='module')
def c():return Custody()


def test_hpke_official_rfc9180_base_vector():
    v=json.loads((Path(__file__).parent/'fixtures/hpke_rfc9180.json').read_text())
    sk=X25519PrivateKey.from_private_bytes(bytes.fromhex(v['skEm']))
    pk=bytes.fromhex(v['pkRm']);enc=bytes.fromhex(v['enc']);info=bytes.fromhex(v['info'])
    key,nonce=hpke.schedule(sk.exchange(X25519PublicKey.from_public_bytes(pk)),enc,pk,info)
    assert key.hex()==v['key'];assert nonce.hex()==v['base_nonce']
    e=v['encryptions'][0]
    assert AESGCM(key).encrypt(nonce,bytes.fromhex(e['pt']),bytes.fromhex(e['aad'])).hex()==e['ct']


def test_td_second_generator_and_integrity(c):
    assert te.GBAR!=te.G and te.H_PEDERSEN not in (te.G,te.GBAR)
    cap=te.seal(c.release_key,b'capsule inner',{'copy':'a'})
    for k in ('abar','a','e','z'):
        bad=copy.deepcopy(cap);bad[k]+=1
        assert not te.check_capsule(c.release_key,bad,{'copy':'a'})
    bad=copy.deepcopy(cap);bad['c']='00'*32
    assert not te.check_capsule(c.release_key,bad,{'copy':'a'})


def test_pedersen_invalid_share_rejected():
    a=[4,7,9];b=[6,3,11]
    C=[pow(te.G,x,te.P)*pow(te.H_PEDERSEN,y,te.P)%te.P for x,y in zip(a,b)]
    A=[pow(te.G,x,te.P) for x in a]
    value=sum(x*2**k for k,x in enumerate(a));blind=sum(x*2**k for k,x in enumerate(b))
    assert te.verify_vss(2,value,blind,C,A)
    assert not te.verify_vss(2,value+1,blind,C,A)
    assert not te.verify_vss(2,value,blind+1,C,A)


def test_stream_delivery_and_chunk_substitution(c,tmp_path):
    src=tmp_path/'model';src.write_bytes(b'actual-model-bytes'*(300000))
    p=issue_file(c,1,src,tmp_path/'transit','stream-test')
    r=c.request(c.members[0],p)
    for m in c.members[1:4]:c.intent(m,r,p)
    inner=c.release(r,p,[c.approval(m,r,p) for m in c.members[1:4]])
    dest=receive_file(c,c.members[0],p,inner,tmp_path/'received')
    assert file_hash(src)==file_hash(dest)
    bad=copy.deepcopy(p);bad['chunks'].reverse()
    with pytest.raises(ValueError):receive_file(c,c.members[0],bad,inner,tmp_path/'wrong')
    assert not (tmp_path/'wrong').exists()


def test_probe_commit_before_disclose_and_one_use(c):
    spec={'probes':[{'text':'a natural question','tokens':[1,2]}]}
    cap=escrow_parameters(c,'escrow-test',spec)
    with pytest.raises(ValueError):disclose_parameters(c,'escrow-test',cap,[1,2,3],'filehash')
    c.ledger.append('evidence_commitment',dict(group='escrow-test',file_sha256='filehash'))
    with pytest.raises(ValueError):disclose_parameters(c,'escrow-test',cap,[1,2],'filehash')
    assert disclose_parameters(c,'escrow-test',cap,[1,2,3],'filehash')==spec
    with pytest.raises(ValueError):disclose_parameters(c,'escrow-test',cap,[1,2,3],'filehash')


def test_journal_restart_merkle_and_witness(c,tmp_path):
    j=Journal(tmp_path/'ledger.jsonl');j.append('test',{'value':1})
    first=j.checkpoint(c.issuer,c.members[:3]);j.append('test',{'value':2})
    final=j.checkpoint(c.issuer,c.members[:3])
    assert verify_checkpoint(final,c.issuer_pk,c.registry,3)
    assert verify_extension(first['head'],final['head'],j.entries)
    j2=Journal(j.path);assert j2.entries==j.entries
    leaves=[leaf(canonical(e)) for e in j.entries]
    assert verify_inclusion(j.entries[0],inclusion(leaves,0),final['head']['root'])
    bad=copy.deepcopy(final);bad['witnesses']=bad['witnesses'][:2]
    assert not verify_checkpoint(bad,c.issuer_pk,c.registry,3)
    assert not verify_extension(final['head'],first['head'],j.entries[:1])


def test_api_quorum_exact_words_and_signed_raw_responses(c):
    now=time.time();target=dict(endpoint='https://test.invalid/model',tls_sha256='test',start=now-1,end=now+60)
    probes=[dict(text='natural text',bit=0)]*3
    c.ledger.append('endpoint_commitment',dict(group='api-test',target=target))
    word,records=collect_quorum(c,target,probes,lambda target,probes,i:[1.,1.,1.],[1,2,3],'api-test')
    assert word==[1]
    bad=copy.deepcopy(records);bad[1]['body']['scores']=[-1.,-1.,-1.]
    bad[1]['signature']=sign(c.members[1].identity,'api-evidence',bad[1]['body'])
    with pytest.raises(ValueError,match='disagree'):verify_quorum(bad,c.registry,3,target,probes,'api-test')
    with pytest.raises(ValueError):verify_quorum(records[:2],c.registry,3,target,probes,'api-test')


def test_anchored_relay_rebuttal(c):
    now=time.time();log=ServedLog(c.members[0],c.ledger)
    probes=[dict(text='a natural question about the model',bit=0),dict(text='different topic and subject',bit=1)]
    log.record(probes[0]['text'],'pirate-client',now);anchor=log.anchor()
    rebut=log.rebut(probes,(now-1,now+1))
    assert verify_rebuttal(rebut,anchor,c.registry,probes,(now-1,now+1))==[0]
    bad=copy.deepcopy(rebut);bad['entries'][0]['client']='someone-else'
    with pytest.raises(ValueError):verify_rebuttal(bad,anchor,c.registry,probes,(now-1,now+1))


def test_live_tls_endpoint_and_pin_rejection(c,tmp_path):
    import torch
    from types import SimpleNamespace
    from tvmc.api_server import Endpoint
    from tvmc.evidence import https_query
    class TinyTestLab:
        def tokens(self,text):return {}
        def last_logits(self,x):return torch.tensor([0.,2.,1.])
    log=ServedLog(c.members[0],c.ledger)
    endpoint=Endpoint(tmp_path/'tls',TinyTestLab(),log)
    try:
        probes=[dict(text='test request',tokens=[1,2],bit=0)]
        assert https_query(endpoint.target,probes,2)==[1.]
        bad=dict(endpoint.target,tls_sha256='00'*32)
        with pytest.raises(ValueError,match='pin'):https_query(bad,probes,2)
        assert len(log.entries)==1
    finally:endpoint.close()


def test_rebuttal_rescore_public_verification(c):
    from tvmc.paper_trace import FileTracer,rescore_rebuttal,judge_rebuttal
    from tvmc.crypto import digest
    tracer=FileTracer(c,[.5]*4,2,'rebut-score-test')
    tracer.enroll('rebut-copy',[1]*4);c.issue(1,b'[1,1,1,1]','rebut-copy')
    tracer.commit_file(b'[1,1,1,1]');original=tracer.trace_file(b'[1,1,1,1]',json.loads,[2,3,4])
    rebut={'validated_log':'unit-fixture'};positions=[0,1,2,3]
    c.ledger.append('verified_rebuttal',dict(group=tracer.group,cid='rebut-copy',erased=positions,digest=digest(rebut)))
    res=rescore_rebuttal(c,tracer,original,'rebut-copy',positions,[2,3,4],digest(rebut))
    assert res['score']==0 and not res['retained']
    bundle=dict(rebuttal=rebut,result=res)
    assert judge_rebuttal(original,bundle,positions,c.ledger.entries,c.registry,c.code_key)
    bad=copy.deepcopy(bundle);bad['result']['retained']=True
    assert not judge_rebuttal(original,bad,positions,c.ledger.entries,c.registry,c.code_key)
