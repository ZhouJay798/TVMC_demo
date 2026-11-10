import copy
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from tvmc.bundle_delivery import make_inner, seal_bundle
from tvmc.crypto import canonical
from tvmc.tardos import Tardos
from tvmc.personal_delivery import replica_seed
from tvmc.trusted_registry import (code_commitment, create_precommit, create_issuance,
    create_precommit_checkpoint, create_checkpoint, verify_checkpoint,
    verify_issuance, verify_recipient_binding, verify_snapshot)
from run_personal_delivery import setup


@pytest.fixture(scope='module')
def signed_delivery():
    gate, admission, (key, _, _, _, _), keys, recipients, _ = setup('a'*64)
    issuer = Ed25519PrivateKey.generate()
    registry = Ed25519PrivateKey.generate()
    witness = Ed25519PrivateKey.generate()
    tardos = Tardos(n=3)
    words = [tardos.encode(replica_seed(admission, r)) for r in recipients]
    pre, openings = create_precommit(gate, admission, recipients, words, registry, tardos)
    pre_challenge = 'b'*64
    pre_checkpoint = create_precommit_checkpoint(pre, registry.public_key(), witness, pre_challenge, now=1000)
    entries = [make_inner(gate, admission, r, f'model-{i}'.encode()) for i, r in enumerate(recipients)]
    _, manifest, signature = seal_bundle(gate, admission, recipients, entries, key, issuer)
    issued = create_issuance(pre, registry.public_key(), manifest, signature, issuer.public_key(),
                             gate, admission, recipients, registry)
    challenge = 'c'*64
    checkpoint = create_checkpoint(pre, issued, registry.public_key(), witness, challenge,
                                   pre_checkpoint, pre_challenge, now=1010)
    return locals()


def validate(f):
    return verify_issuance(f['pre'], f['issued'], f['registry'].public_key(), f['manifest'],
                           f['signature'], f['issuer'].public_key(), f['gate'], f['admission'], f['recipients'])


def test_signed_chain_and_unique_recipient_binding(signed_delivery):
    f=signed_delivery
    body=validate(f)
    assert verify_checkpoint(f['checkpoint'],f['witness'].public_key(),f['issued'],
                             f['registry'].public_key(),f['challenge'],now=1011)
    for i, record in enumerate(f['recipients']):
        rid=f"C-E0-U{i+1}"
        entry=verify_recipient_binding(body,rid,i+1,bytes.fromhex(record['public_key_der']))
        assert entry['code_commitment']==code_commitment(rid,f['words'][i],f['openings'][rid])
    with pytest.raises(ValueError):
        verify_recipient_binding(body,'C-E0-U2',1,bytes.fromhex(f['recipients'][0]['public_key_der']))


def test_rejects_tamper_wrong_trust_anchors_and_swapped_manifest(signed_delivery):
    f=signed_delivery
    changed=copy.deepcopy(f['issued']);changed['body']['entries'][0]['recipient']=2
    with pytest.raises(ValueError):verify_snapshot(changed,f['registry'].public_key())
    with pytest.raises(ValueError):verify_snapshot(f['issued'],Ed25519PrivateKey.generate().public_key())
    altered=copy.deepcopy(f['manifest']);altered['entries'][0]['context']['recipient']=2
    with pytest.raises(ValueError):
        verify_issuance(f['pre'],f['issued'],f['registry'].public_key(),altered,f['signature'],
                        f['issuer'].public_key(),f['gate'],f['admission'],f['recipients'])
    with pytest.raises(ValueError):
        verify_issuance(f['pre'],f['issued'],f['registry'].public_key(),f['manifest'],f['signature'],
                        Ed25519PrivateKey.generate().public_key(),f['gate'],f['admission'],f['recipients'])


def test_signed_but_wrong_registry_mapping_is_rejected(signed_delivery):
    f=signed_delivery
    changed=copy.deepcopy(f['issued']);changed['body']['entries'][0]['recipient']=2
    changed['signature']=f['registry'].sign(canonical(changed['body'])).hex()
    with pytest.raises(ValueError):
        verify_issuance(f['pre'],changed,f['registry'].public_key(),f['manifest'],f['signature'],
                        f['issuer'].public_key(),f['gate'],f['admission'],f['recipients'])


def test_rejects_rollback_expiry_wrong_challenge_and_witness(signed_delivery):
    f=signed_delivery
    for snapshot,challenge,now in [(f['pre'],f['challenge'],1011),
                                   (f['issued'],'d'*64,1011),
                                   (f['issued'],f['challenge'],1310)]:
        with pytest.raises(ValueError):
            verify_checkpoint(f['checkpoint'],f['witness'].public_key(),snapshot,
                              f['registry'].public_key(),challenge,now=now)
    with pytest.raises(ValueError):
        verify_checkpoint(f['checkpoint'],Ed25519PrivateKey.generate().public_key(),f['issued'],
                          f['registry'].public_key(),f['challenge'],now=1011)
    with pytest.raises(ValueError):
        create_checkpoint(f['pre'],f['issued'],f['registry'].public_key(),f['witness'],f['challenge'],
                          f['pre_checkpoint'],'wrong-challenge',now=1010)


def test_duplicate_codewords_rejected(signed_delivery):
    f=signed_delivery
    with pytest.raises(ValueError):
        create_precommit(f['gate'],f['admission'],f['recipients'],[f['words'][0]]*3,f['registry'],f['tardos'])


def test_wrong_codeword_and_signed_parameter_substitution_rejected(signed_delivery):
    f=signed_delivery
    swapped=[f['words'][1],f['words'][0],f['words'][2]]
    with pytest.raises(ValueError):
        create_precommit(f['gate'],f['admission'],f['recipients'],swapped,f['registry'],f['tardos'])
    changed=copy.deepcopy(f['issued'])
    changed['body']['fingerprint_scheme']['threshold']-=1
    changed['signature']=f['registry'].sign(canonical(changed['body'])).hex()
    with pytest.raises(ValueError):
        verify_issuance(f['pre'],changed,f['registry'].public_key(),f['manifest'],f['signature'],
                        f['issuer'].public_key(),f['gate'],f['admission'],f['recipients'])
