"""Adversarial requests must be rejected before any secret-share operation."""
import copy
import json
import pytest
from tvmc.paper_workflow import Custody, sha
from tvmc.paper_trace import FileTracer, judge_file
from tvmc.score_approval import ScoreApprover, score_ciphertext
from tvmc import paper_threshold as te


@pytest.fixture(scope='module')
def prepared():
    c=Custody();t=FileTracer(c,[.5]*4,2,'guard-test')
    for i,w in [(1,[1,0,1,0]),(2,[0,1,0,1])]:
        t.enroll(str(i),w);c.issue(i,json.dumps(w).encode(),str(i))
    artifact=b'[1,0,1,0]';t.commit_file(artifact)
    c.ledger.append('group_disclosed_and_retired',dict(parameters=t.params,salt=t.salt))
    entries=copy.deepcopy(c.ledger.entries);head=entries[-1]['hash'];word=json.loads(artifact)
    req=dict(group=t.group,kid=c.code_key.kid,head=head,evidence=sha(artifact),observed=word,
             scores=[dict(cid=cid,**score_ciphertext(t.params,word,cts)) for cid,cts in t.codebook.items()])
    c.ledger.append('score_request',req)
    entries=copy.deepcopy(c.ledger.entries);head=entries[-1]['hash']
    return c,t,artifact,entries,head,req


def approver(c,i=1):
    return ScoreApprover(c.code_key,c.code_shares[i-1],c.members[i-1].identity,c.issuer_pk,c.registry,json.loads)


@pytest.mark.parametrize('attack',['bit_ciphertext','score_flip','omit_copy','wrong_word','wrong_key','wrong_head','wrong_evidence','tampered_ledger'])
def test_reject_before_any_partial(prepared,monkeypatch,attack):
    c,t,artifact,entries,head,req=prepared
    entries=copy.deepcopy(entries);req=copy.deepcopy(req)
    if attack=='bit_ciphertext':req['scores'][0].update({k:t.codebook['1'][0][k] for k in ('a','b')})
    elif attack=='score_flip':req['scores'][0]['b']=req['scores'][0]['b']*te.G%te.P
    elif attack=='omit_copy':req['scores'].pop()
    elif attack=='wrong_word':req['observed']=[0]*4
    elif attack=='wrong_key':req['kid']=c.release_key.kid
    elif attack=='wrong_head':req['head']='0'*64
    elif attack=='wrong_evidence':artifact=b'[0,0,0,0]'
    elif attack=='tampered_ledger':entries[-2]['body']['parameters']['biases'][0]=.4
    if attack in ('bit_ciphertext','score_flip','omit_copy','wrong_word','wrong_key'):
        from tvmc.crypto import digest
        entries[-1]['body']=copy.deepcopy(req)
        entries[-1]['hash']=digest({k:v for k,v in entries[-1].items() if k!='hash'})
        head=entries[-1]['hash']
    def forbidden(*a,**kw):pytest.fail('private-share operation before validation')
    monkeypatch.setattr(te,'partial',forbidden)
    with pytest.raises(ValueError):approver(c).approve(req,entries,head,artifact)


def test_batch_decrypt_and_replay(prepared,monkeypatch):
    c,t,artifact,entries,head,req=prepared
    guards=[approver(c,i) for i in (1,2,3)]
    replies=[g.approve(req,entries,head,artifact) for g in guards]
    for index,row in enumerate(req['scores']):
        ctx=['score',t.group,sha(artifact),req['observed'],row['cid'],row['a'],row['b']]
        proofs=[r['responses'][index]['proof'] for r in replies]
        shared=te.combine(c.code_key,row['a'],proofs,ctx)
        assert row['b']*pow(shared,-1,te.P)%te.P==pow(te.G,(4 if index==0 else -4)*1024,te.P)
    with pytest.raises(ValueError,match='already used'):guards[0].approve(req,entries,head,artifact)


def test_api_signed_evidence_to_public_judge(prepared,monkeypatch):
    import time
    from tvmc.crypto import canonical,digest
    from tvmc.paper_trace import escrow_parameters,disclose_parameters,judge_endpoint
    from tvmc.evidence import collect_quorum,verify_quorum
    c=prepared[0];group='guard-api'
    probes=[dict(text='fixture',tokens=[1,2],bit=k) for k in range(4) for _ in range(3)]
    spec=dict(kind='signed-api-logit-difference',probe_hash=digest(probes),repeats=3,delta=1.)
    t=FileTracer(c,[.5]*4,2,group,reader_spec=spec)
    cap=escrow_parameters(c,group,dict(probes=probes,repeats=3,delta=1.))
    for i,w in [(1,[1,0,1,0]),(2,[0,1,0,1])]:
        cid='api-'+str(i);t.enroll(cid,w);c.issue(i,json.dumps(w).encode(),cid)
    now=time.time();target=dict(endpoint='https://fixture.invalid',tls_sha256='fixture',start=now-1,end=now+120)
    c.ledger.append('endpoint_commitment',dict(group=group,target=target))
    artifact=canonical(target);t.commit_file(artifact)
    disclose_parameters(c,group,cap,[2,3,4],sha(artifact))
    word,records=collect_quorum(c,target,probes,lambda *a:[1.]*3+[-1.]*3+[1.]*3+[-1.]*3,[2,3,4],group)
    reader=lambda _:verify_quorum(records,c.registry,3,target,probes,group)
    result=t.trace_file(artifact,reader,[2,3,4])
    assert result['recipients']==[1]
    assert result['scores']=={'api-1':4.,'api-2':-4.}
    assert judge_endpoint(result,target,probes,records,c,c.ledger.entries[-1]['hash'])
    # Public audit rejects an altered approval even if the attacker rebuilds the chain.
    saved=copy.deepcopy(c.ledger.entries)
    for e in c.ledger.entries:
        if e['kind']=='score_approval' and e['body']['approval']['request_hash']==result['request_hash']:
            e['body']['signature']='00'*64;break
    prev='0'*64
    for e in c.ledger.entries:
        e['prev']=prev;e['hash']=digest({k:v for k,v in e.items() if k!='hash'});prev=e['hash']
    assert not judge_endpoint(result,target,probes,records,c,prev)
    c.ledger.entries=saved
    request_entry=next(e for e in saved if e['kind']=='score_request' and e['body']['group']==group)
    request=copy.deepcopy(request_entry['body'])
    snapshot=copy.deepcopy(saved[:request_entry['seq']+1])
    collected=next(e for e in snapshot if e['kind']=='endpoint_evidence' and e['body']['group']==group)
    collected['body']['records'][0]['body']['scores'][0]=-100.
    prev='0'*64
    for e in snapshot:
        if e['kind']=='score_request' and e['body']['group']==group:
            request['head']=prev;e['body']=copy.deepcopy(request)
        e['prev']=prev;e['hash']=digest({k:v for k,v in e.items() if k!='hash'});prev=e['hash']
    def forbidden(*a,**kw):pytest.fail('decrypted before validating API signatures')
    monkeypatch.setattr(te,'partial',forbidden)
    with pytest.raises(ValueError,match='collector signature'):
        approver(c).approve(request,snapshot,prev,artifact)
