import copy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from run_personal_delivery import setup
from tvmc.crypto import prove
from tvmc.process_recovery import reconstruct_submissions
from tvmc.reconstruction import request_context, verify_receipt


@pytest.fixture(scope='module')
def material():
    _, _, (key, public, shares, _, key_context), _, _, _ = setup('a' * 64)
    signer = Ed25519PrivateKey.generate()
    context = request_context(public, key_context, 'b' * 64, {'demo': 1}, 'c' * 64)
    submissions = [{'i': share.i, 'value': share.value, 'proof': prove(public, share, context)}
                   for share in shares]
    return key, public, context, signer, submissions


def test_reconstructs_without_identity_secrets_in_submissions(material):
    key, public, context, signer, submissions = material
    assert all('gamma' not in submission for submission in submissions)
    restored, receipt = reconstruct_submissions(public, submissions, context, signer)
    assert restored == key
    assert verify_receipt(receipt, public, context, signer.public_key())['verified_participants'] == [1, 2, 3]
    assert all('value' not in proof and 'gamma' not in proof for proof in receipt['record']['proofs'])


def test_rejects_missing_duplicate_tampered_and_cross_request(material):
    _, public, context, signer, submissions = material
    wrong_value = copy.deepcopy(submissions)
    wrong_value[0]['value'] += 1
    wrong_proof = copy.deepcopy(submissions)
    wrong_proof[0]['proof']['zf'] += 1
    wrong_context = dict(context, request_nonce='d' * 64)
    for items, ctx in ((submissions[:2], context), ([submissions[0]] * 3, context),
                       (wrong_value, context), (wrong_proof, context),
                       (submissions, wrong_context)):
        with pytest.raises(ValueError):
            reconstruct_submissions(public, items, ctx, signer)
