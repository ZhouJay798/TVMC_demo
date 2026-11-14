import copy
from datetime import timedelta
import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization
from tvmc import pki
from tvmc.paper_workflow import Custody


@pytest.fixture(scope='module')
def custody():return Custody()


def test_all_three_certificate_roles(custody):
    c=custody
    assert c.issuer_pk==c.issuer.public_key().public_bytes_raw()
    for m in c.members:
        assert pki.member_key(c.registry,m.i)==m.identity.public_key().public_bytes_raw()
        assert pki.member_key(c.registry,m.i,'receiving')==m.receiving.public_key().public_bytes_raw()
    assert 'PRIVATE KEY' not in str(c.registry.trust.export())


@pytest.mark.parametrize('purpose',['identity','receiving'])
@pytest.mark.parametrize('attack',['raw_key','other_member','foreign_ca','missing','signature'])
def test_reject_member_key_substitution(custody,purpose,attack):
    reg=copy.deepcopy(custody.registry);r=reg[1];field=purpose+'_certificate'
    if attack=='raw_key':r[purpose]='00'*32
    elif attack=='other_member':r[field]=reg[2][field];r[purpose]=reg[2][purpose]
    elif attack=='foreign_ca':
        key=(Ed25519PrivateKey if purpose=='identity' else X25519PrivateKey).generate()
        r[field]=pki.DemoCA().issue('U1',key.public_key(),purpose);r[purpose]=key.public_key().public_bytes_raw().hex()
    elif attack=='missing':del r[field]
    else:
        cert=x509.load_pem_x509_certificate(r[field].encode());der=cert.public_bytes(serialization.Encoding.DER)
        r[field]=pki.pem(x509.load_der_x509_certificate(der[:-1]+bytes([der[-1]^1])))
    with pytest.raises(ValueError):pki.member_key(reg,1,purpose)


def test_issuer_substitution_and_wrong_usage(custody):
    with pytest.raises(ValueError):pki.issuer_key(custody.registry,b'0'*32)
    trust=copy.deepcopy(custody.registry.trust)
    trust.issuer_certificate=custody.registry[1]['identity_certificate']
    with pytest.raises(ValueError):trust.validate(trust.issuer_certificate,'distributor','issuer')
    with pytest.raises(ValueError):trust.validate(custody.registry[1]['identity_certificate'],'U1','receiving')


def test_revoke_and_crl_rollback(custody):
    reg=copy.deepcopy(custody.registry);ca=custody.ca
    old=reg.trust.crl_pem
    # Restore CA state so module fixture remains usable for other cases.
    revoked,number=ca.revoked.copy(),ca.number
    try:
        new=ca.revoke(reg[1]['receiving_certificate']);reg.trust.update_crl(new)
        with pytest.raises(ValueError):pki.member_key(reg,1,'receiving')
        assert pki.member_key(reg,2,'receiving')
        with pytest.raises(ValueError):reg.trust.update_crl(old)
    finally:ca.revoked,ca.number=revoked,number


def test_expired_certificate_and_stale_crl(custody,monkeypatch):
    reg=copy.deepcopy(custody.registry);future=pki.now()+timedelta(days=31)
    monkeypatch.setattr(pki,'now',lambda:future)
    with pytest.raises(ValueError,match='CRL'):pki.member_key(reg,1)
    reg.trust.crl_pem=custody.ca.crl()
    with pytest.raises(ValueError,match='expired'):pki.member_key(reg,1)


def test_issue_rejects_substituted_receiver_before_hpke(custody,monkeypatch):
    c=custody;saved=copy.deepcopy(c.registry[1]);c.registry[1]['receiving']='00'*32
    def forbidden(*a,**kw):pytest.fail('used unverified recipient key')
    monkeypatch.setattr('tvmc.paper_workflow.hpke.seal',forbidden)
    try:
        with pytest.raises(ValueError):c.issue(1,b'test')
    finally:c.registry[1]=saved


def test_untrusted_plain_dictionary_cannot_bypass_ca(custody):
    with pytest.raises(ValueError):pki.member_key(dict(custody.registry),1)


def test_revoked_issuer_stops_service(custody):
    reg=copy.deepcopy(custody.registry);ca=custody.ca
    saved=(ca.revoked.copy(),ca.number)
    try:
        reg.trust.update_crl(ca.revoke(reg.trust.issuer_certificate))
        with pytest.raises(ValueError):pki.issuer_key(reg)
    finally:ca.revoked,ca.number=saved
