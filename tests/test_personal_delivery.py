import copy
import pytest
from cryptography.exceptions import InvalidTag,InvalidSignature
from run_identity_gate_experiment import enrolled,new_session,key_material
from tvmc.crypto import recover
from tvmc.personal_delivery import new_key,bind_recipient,verify_recipient,seal_copy,open_copy,context_for

@pytest.fixture(scope='module')
def setup():
    identities,_,gate=enrolled();adm,_,_=new_session(gate,identities,'C-E0',[1,2,3],'a'*64)
    material=key_material(gate,identities,adm,3);keys=[new_key() for _ in range(3)]
    records=[bind_recipient(gate,adm,i,identities[i-1],keys[i-1]) for i in (1,2,3)]
    packages=[seal_copy(gate,adm,r,material[0],b'model'+bytes([i])) for i,r in enumerate(records)]
    return gate,adm,material,keys,records,packages

def test_threshold_then_personal_keys(setup):
    _,_,(k,pub,shares,proofs,ctx),keys,_,packages=setup
    with pytest.raises(ValueError):recover(pub,shares[:2],proofs[:2],ctx)
    assert recover(pub,shares,proofs,ctx)==k
    for i,package in enumerate(packages):
        assert open_copy(k,keys[i],package,package['context'])==b'model'+bytes([i])
        for j in range(3):
            if i!=j:
                with pytest.raises(ValueError):open_copy(k,keys[j],package,package['context'])
        with pytest.raises(InvalidTag):open_copy(k+1,keys[i],package,package['context'])

@pytest.mark.parametrize('field,value',[('recipient',2),('sid','other'),('epoch',1),('replica_id','other')])
def test_context_changes_rejected(setup,field,value):
    _,_,material,keys,_,packages=setup;p=copy.deepcopy(packages[0]);p['context'][field]=value
    # Even accepting the edited header cannot authenticate the old wrapper.
    with pytest.raises(InvalidTag):open_copy(material[0],keys[0],p,p['context'])

def test_cipher_and_wrapper_tampering(setup):
    _,_,material,keys,_,packages=setup
    for field in ('ciphertext','wrapped_key'):
        p=copy.deepcopy(packages[0]);p[field]=p[field][:-1]+bytes([p[field][-1]^1])
        with pytest.raises(InvalidTag):open_copy(material[0],keys[0],p,p['context'])

def test_key_binding_rejects_swapped_public_key(setup):
    gate,adm,_,_,records,_=setup;r=copy.deepcopy(records[0]);r['public_key_der']=records[1]['public_key_der']
    with pytest.raises((ValueError,InvalidSignature)):verify_recipient(gate,adm,r)
