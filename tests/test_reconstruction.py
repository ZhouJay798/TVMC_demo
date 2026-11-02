import copy
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from tvmc.crypto import canonical, scalar, prove
from tvmc.epochs import create
from tvmc.reconstruction import request_context,reconstruct,verify_receipt

@pytest.fixture(scope='module')
def fixture():
    key,s=create('C',0,[1,2,3],[scalar() for _ in range(5)])
    signer=Ed25519PrivateKey.generate()
    context=request_context(s.public,s.context(),'a'*64,{'entries':[]},'b'*64)
    shares=[x.share for x in s.shares]
    proofs=[prove(s.public,share,context) for share in shares]
    recovered,receipt=reconstruct(s.public,shares,proofs,context,signer)
    assert recovered==key
    return s,signer,context,receipt

def test_roundtrip_and_replay(fixture):
    s,signer,context,receipt=fixture;seen=set()
    assert verify_receipt(receipt,s.public,context,signer.public_key(),seen)['verified_participants']==[1,2,3]
    with pytest.raises(ValueError):verify_receipt(receipt,s.public,context,signer.public_key(),seen)

@pytest.mark.parametrize('field',['request_nonce','bundle_hash','manifest_hash','session'])
def test_context_substitution(fixture,field):
    s,signer,context,receipt=fixture;changed=copy.deepcopy(context);changed[field]='other'
    with pytest.raises(ValueError):verify_receipt(receipt,s.public,changed,signer.public_key())

def test_tampered_proofs_even_with_valid_service_signature(fixture):
    s,signer,context,receipt=fixture;bad=copy.deepcopy(receipt)
    bad['record']['proofs'][0]['zf']+=1
    bad['signature']=signer.sign(canonical(bad['record'])).hex()
    with pytest.raises(ValueError):verify_receipt(bad,s.public,context,signer.public_key())

def test_signature_and_missing_shares(fixture):
    s,signer,context,receipt=fixture
    with pytest.raises(InvalidSignature):verify_receipt(receipt,s.public,context,Ed25519PrivateKey.generate().public_key())
    with pytest.raises(ValueError):reconstruct(s.public,[x.share for x in s.shares[:2]],[],context,signer)


def test_reconstruction_rejects_invalid_or_replayed_submissions(fixture):
    s,signer,context,_=fixture
    shares=[x.share for x in s.shares]
    proofs=[prove(s.public,share,context) for share in shares]
    for invalid in (proofs[:-1], list(reversed(proofs))):
        with pytest.raises(ValueError):reconstruct(s.public,shares,invalid,context,signer)
    bad=copy.deepcopy(proofs);bad[0]['zf']+=1
    with pytest.raises(ValueError):reconstruct(s.public,shares,bad,context,signer)
    other=copy.deepcopy(context);other['request_nonce']='c'*64
    with pytest.raises(ValueError):reconstruct(s.public,shares,proofs,other,signer)
