import copy
import pytest
from cryptography.exceptions import InvalidTag,InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from run_personal_delivery import setup
from tvmc.bundle_delivery import make_inner,seal_bundle,open_bundle,open_inner

@pytest.fixture(scope='module')
def fixture():
    gate,adm,material,keys,records,_=setup('a'*64);issuer=Ed25519PrivateKey.generate();k=material[0]
    items=[make_inner(gate,adm,r,b'model'+bytes([i])) for i,r in enumerate(records)]
    outer,manifest,sig=seal_bundle(gate,adm,records,items,k,issuer)
    return gate,adm,k,keys,records,issuer,outer

def test_own_cross_and_wrong_session_key(fixture):
    gate,adm,k,keys,records,issuer,outer=fixture
    with pytest.raises(InvalidTag):open_bundle(gate,adm,records,outer,k+1,issuer.public_key())
    entries,_=open_bundle(gate,adm,records,outer,k,issuer.public_key())
    for i,e in enumerate(entries):
        assert open_inner(e,keys[i])==b'model'+bytes([i])
        for j in range(3):
            if i!=j:
                with pytest.raises(ValueError):open_inner(e,keys[j])

def test_ciphertext_and_issuer_tamper_rejected(fixture):
    gate,adm,k,keys,records,issuer,outer=fixture
    changed=outer[:-1]+bytes([outer[-1]^1])
    with pytest.raises(InvalidTag):open_bundle(gate,adm,records,changed,k,issuer.public_key())
    with pytest.raises(InvalidSignature):open_bundle(gate,adm,records,outer,k,Ed25519PrivateKey.generate().public_key())

def test_key_substitution_and_wrong_session_rejected(fixture):
    gate,adm,k,keys,records,issuer,outer=fixture
    changed=copy.deepcopy(records);changed[0]['public_key_der']=changed[1]['public_key_der']
    with pytest.raises((ValueError,InvalidSignature)):open_bundle(gate,adm,changed,outer,k,issuer.public_key())
    from dataclasses import replace
    with pytest.raises(ValueError):open_bundle(gate,replace(adm,sid='other'),records,outer,k,issuer.public_key())

def test_submitted_proofs_recover_and_open_persisted_bundle(tmp_path):
    from tvmc.crypto import prove
    from tvmc.reconstruction import request_context,reconstruct,verify_receipt
    from tvmc.personal_delivery import sha
    gate,adm,(key,pub,shares,_,ctx),keys,records,_=setup('a'*64)
    issuer=Ed25519PrivateKey.generate()
    items=[make_inner(gate,adm,r,b'model'+bytes([i])) for i,r in enumerate(records)]
    outer,manifest,_=seal_bundle(gate,adm,records,items,key,issuer)
    path=tmp_path/'outer_bundle.aesgcm';path.write_bytes(outer)
    context=request_context(pub,ctx,sha(outer),manifest,'b'*64)
    proofs=[prove(pub,share,context) for share in shares]
    bad=copy.deepcopy(proofs);bad[0]['zf']+=1
    with pytest.raises(ValueError):reconstruct(pub,shares,bad,context,issuer)
    recovered,receipt=reconstruct(pub,shares,proofs,context,issuer)
    assert verify_receipt(receipt,pub,context,issuer.public_key())['verified_participants']==[1,2,3]
    entries,_=open_bundle(gate,adm,records,path.read_bytes(),recovered,issuer.public_key())
    assert [open_inner(entry,keys[i]) for i,entry in enumerate(entries)]==[b'model\x00',b'model\x01',b'model\x02']
