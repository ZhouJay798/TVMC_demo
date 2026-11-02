import copy
import pytest
from tvmc.crypto import G,P,scalar
from tvmc.identity_gate import IdentityGate,enrollment_context,identity_proof

@pytest.fixture(scope='module')
def setup():
    secrets=[scalar() for _ in range(5)]
    records=[{'i':i,'commitment':pow(G,secrets[i-1],P),'proof':identity_proof(i,secrets[i-1],enrollment_context(i))} for i in range(1,6)]
    return secrets,records

def request(gate,secrets,sid,participants,nonce='fresh'):
    context=gate.context(sid,0,sorted(participants),'a'*64,nonce)
    return [identity_proof(i,secrets[i-1],context) for i in participants]

@pytest.mark.parametrize('ids',[(1,2),(2,3,4),(1,2,3,4,5)])
def test_admission(setup,ids):
    secrets,records=setup;gate=IdentityGate(records);proofs=request(gate,secrets,'s',ids)
    admission=gate.register('s',0,ids,'a'*64,'fresh',proofs)
    assert admission.participants==tuple(ids) and len(gate.sessions)==1

@pytest.mark.parametrize('fault',['insufficient','duplicate','missing','tampered','replay','outsider'])
def test_bad_requests_do_not_register(setup,fault):
    secrets,records=setup;gate=IdentityGate(records);ids=[1,2];proofs=request(gate,secrets,'s',ids)
    if fault=='insufficient':ids=[1];proofs=proofs[:1]
    if fault=='duplicate':ids=[1,1];proofs=[proofs[0],proofs[0]]
    if fault=='missing':proofs.pop()
    if fault=='tampered':proofs[0]['z']+=1
    if fault=='replay':proofs=request(gate,secrets,'other',ids)
    if fault=='outsider':ids=[1,6]
    with pytest.raises(ValueError):gate.register('s',0,ids,'a'*64,'fresh',proofs)
    assert not gate.sessions

def test_preregistration_rejects_bad_proof_and_duplicate(setup):
    _,records=setup;bad=copy.deepcopy(records);bad[0]['proof']['z']+=1
    with pytest.raises(ValueError):IdentityGate(bad)
    with pytest.raises(ValueError):IdentityGate(records+[records[0]])


def test_delivery_stages_require_real_admission(setup):
    from dataclasses import replace
    secrets,records=setup;gate=IdentityGate(records);proofs=request(gate,secrets,'s',[1,2])
    admitted=gate.register('s',0,[1,2],'a'*64,'fresh',proofs)
    assert gate.require_admitted(admitted) is admitted
    with pytest.raises(ValueError):gate.require_admitted(replace(admitted,sid='unregistered'))
    with pytest.raises(ValueError):IdentityGate(records).require_admitted(admitted)

@pytest.mark.parametrize('count',[2,3,5])
def test_one_share_per_registered_custodian_and_selected_threshold(count):
    from run_identity_gate_experiment import key_material,enrolled,new_session
    from tvmc.crypto import recover
    participants=list(range(1,count+1))
    selected=count
    identities,_,gate=enrolled();admission,_,_=new_session(gate,identities,'test',participants,'a'*64)
    key,public,shares,proofs,context=key_material(gate,identities,admission,selected)
    assert len(shares)==count and {s.i for s in shares}==set(participants)
    assert recover(public,shares[:selected],proofs[:selected],context)==key
    with pytest.raises(ValueError):recover(public,shares[:selected-1],proofs[:selected-1],context)


@pytest.mark.parametrize('count',[3,5])
def test_reduced_threshold_rejected(count):
    from run_identity_gate_experiment import key_material,enrolled,new_session
    from tvmc.epochs import EpochPublic
    identities,_,gate=enrolled();admission,_,_=new_session(gate,identities,'test',list(range(1,count+1)),'a'*64)
    with pytest.raises(ValueError):key_material(gate,identities,admission,2)
    with pytest.raises(ValueError):EpochPublic((1,1),tuple(gate.registry[i] for i in range(1,6)),5,2,tuple(admission.participants))
