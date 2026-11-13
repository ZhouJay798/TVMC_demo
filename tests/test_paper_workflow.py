import copy
import itertools
import json
import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.exceptions import InvalidTag
from tvmc import paper_hpke as hpke
from tvmc import paper_threshold as te
from tvmc.paper_workflow import Custody,Member
from tvmc.paper_trace import encrypt_bit,verify_bit,FileTracer,judge_file


@pytest.fixture(scope='module')
def consortium():return Custody()


def prepared(c):
    p=c.issue(1,b'fingerprinted fixture')
    r=c.request(c.members[0],p)
    for m in c.members[1:4]:c.intent(m,r,p)
    a=[c.approval(m,r,p) for m in c.members[1:4]]
    return p,r,a


def test_hpke_roundtrip_and_context_recipient_tamper():
    a=X25519PrivateKey.generate();b=X25519PrivateKey.generate()
    sealed=hpke.seal(a.public_key().public_bytes_raw(),b'secret',b'copy-A')
    assert hpke.open(a,sealed,b'copy-A')==b'secret'
    for key,blob,info in [(b,sealed,b'copy-A'),(a,sealed,b'copy-B'),
                          (a,sealed[:-1]+bytes([sealed[-1]^1]),b'copy-A')]:
        with pytest.raises(InvalidTag):hpke.open(key,blob,info)


def test_every_three_combine_without_secret_reconstruction(consortium):
    c=consortium;pub=c.release_key;payload=b'inner ciphertext';cap=te.seal(pub,payload,'copy')
    proofs=[te.partial(pub,s,cap['a'],['test',cap]) for s in c.release_shares]
    for subset in itertools.combinations(proofs,3):
        assert te.open_capsule(pub,cap,subset,'copy',['test',cap])==payload
    for bad in [proofs[:2],[proofs[0]]*3]:
        with pytest.raises(ValueError):te.open_capsule(pub,cap,bad,'copy',['test',cap])
    with pytest.raises(ValueError):te.open_capsule(pub,cap,proofs[:3],'copy',['other'])
    assert not te.check_capsule(pub,dict(cap,label='other'),'other')
    assert not te.check_capsule(c.code_key,cap,'copy')


def test_public_share_key_and_purpose_binding(consortium):
    c=consortium
    assert c.release_key.pk!=c.code_key.pk
    for share in c.release_shares:assert pow(te.G,share.value,te.P)==c.release_key.vk(share.i)
    with pytest.raises(ValueError):te.partial(c.code_key,c.release_shares[0],te.G,'wrong-purpose')


def test_two_phase_release_recipient_and_replay(consortium):
    c=consortium;p=c.issue(1,b'model');r=c.request(c.members[0],p)
    with pytest.raises(ValueError):c.approval(c.members[1],r,p)
    for m in c.members[1:4]:c.intent(m,r,p)
    a=[c.approval(m,r,p) for m in c.members[1:4]]
    with pytest.raises(ValueError):c.release(r,p,a[:2])
    with pytest.raises(ValueError):c.release(r,p,[a[0]]*3)
    bad=copy.deepcopy(a);bad[0]['proof']['d']=te.G
    with pytest.raises(ValueError):c.release(r,p,bad)
    inner=c.release(r,p,a)
    assert c.receive(c.members[0],p,inner)==b'model'
    with pytest.raises(ValueError):c.receive(c.members[1],p,inner)
    with pytest.raises(ValueError):c.release(r,p,a)


def test_reject_tampered_package_and_identity_substitution(consortium):
    c=consortium;p=c.issue(1,b'A'*(4*1024*1024+1))
    for field in ['chunks','capsule','manifest']:
        bad=copy.deepcopy(p)
        if field=='chunks':bad[field].reverse()
        if field=='capsule':bad[field]['ct']='00'
        if field=='manifest':bad[field]['message']['recipient']=2
        with pytest.raises(ValueError):c.request(c.members[0],bad)
    with pytest.raises(ValueError):c.request(c.members[1],p)
    with pytest.raises(ValueError):c.request(Member(1),p)


def test_approval_cannot_move_to_another_request(consortium):
    c=consortium;p,r,a=prepared(c);r2=c.request(c.members[0],p)
    for m in c.members[1:4]:c.intent(m,r2,p)
    with pytest.raises(ValueError):c.release(r2,p,a)


def test_bit_proof_context_and_ciphertext_tamper(consortium):
    p=consortium.code_key
    for bit in (0,1):
        ct=encrypt_bit(p,bit,['cid',1]);assert verify_bit(p,ct,['cid',1])
        assert not verify_bit(p,ct,['cid',2])
        bad=copy.deepcopy(ct);bad['b']=bad['b']*te.G%te.P
        assert not verify_bit(p,bad,['cid',1])


def test_encrypted_scores_evidence_binding_and_single_use(consortium):
    c=consortium;t=FileTracer(c,[.5]*8,4,'test-group')
    for i,word in [(1,[1]*8),(2,[0]*8)]:
        cid=f'trace-unit-{i}';t.enroll(cid,word);c.issue(i,json.dumps(word).encode(),cid)
    blob=json.dumps([1]*8).encode()
    with pytest.raises(ValueError):t.trace_file(blob,json.loads,[1,2,3])
    t.commit_file(blob)
    with pytest.raises(ValueError):t.trace_file(b'[]',json.loads,[1,2,3])
    with pytest.raises(ValueError):t.trace_file(blob,json.loads,[1,1,2])
    result=t.trace_file(blob,json.loads,[1,2,3])
    assert result['scores']=={'trace-unit-1':8.,'trace-unit-2':-8.}
    assert result['recipients']==[1]
    args=(blob,json.loads,c.ledger.entries,c.ledger.entries[-1]['hash'],c.issuer_pk,c.registry,c.code_key)
    assert judge_file(result,*args)
    bad=copy.deepcopy(result);bad['recipients']=[2]
    assert not judge_file(bad,*args)
    with pytest.raises(ValueError):t.trace_file(blob,json.loads,[1,2,3])


def test_ledger_tamper_and_pinned_checkpoint(consortium):
    ledger=copy.deepcopy(consortium.ledger);head=ledger.entries[-1]['hash']
    assert ledger.check(head)
    ledger.entries.pop();assert not ledger.check(head)
    ledger.entries[0]['body']['registration']['i']=99
    assert not ledger.check()


def test_refresh_preserves_encryption_key_but_changes_share_epoch(consortium):
    pub=consortium.release_key;shares=consortium.release_shares
    new,new_shares=te.refresh(pub,shares)
    assert new.pk==pub.pk and new.kid!=pub.kid and new.epoch==pub.epoch+1
    assert all(a.value!=b.value for a,b in zip(shares,new_shares))
    for share in new_shares:assert pow(te.G,share.value,te.P)==new.vk(share.i)
    with pytest.raises(ValueError):te.partial(new,shares[0],te.G,'old-share')
    # For existing ciphertexts the service needs an explicit epoch rebind policy;
    # this test only verifies the unchanged algebraic DH key under refreshed shares.
    base=pow(te.G,17,te.P)
    old=te.combine(pub,base,[te.partial(pub,s,base,'old') for s in shares[:3]],'old')
    fresh=te.combine(new,base,[te.partial(new,s,base,'new') for s in new_shares[:3]],'new')
    assert old==fresh
