import copy
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from audit_signed_registry import audit_directory
from run_personal_delivery import setup
from tvmc.bundle_delivery import make_inner, seal_bundle
from tvmc.crypto import prove
from tvmc.personal_delivery import replica_seed, sha
from tvmc.reconstruction import reconstruct, request_context
from tvmc.tardos import Tardos
from tvmc.trusted_registry import (
    create_checkpoint, create_issuance, create_precommit, create_precommit_checkpoint,
)


def dump(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def audit_run(tmp_path):
    gate, admission, (key, public, shares, _, keyctx), _, recipients, reg = setup('a' * 64)
    issuer = Ed25519PrivateKey.generate()
    registry = Ed25519PrivateKey.generate()
    witness = Ed25519PrivateKey.generate()
    tardos = Tardos(n=3)
    words = [tardos.encode(replica_seed(admission, r)) for r in recipients]
    pre, openings = create_precommit(gate, admission, recipients, words, registry, tardos)
    prior = create_precommit_checkpoint(pre, registry.public_key(), witness, 'b' * 64, now=1000)
    entries = [make_inner(gate, admission, r, f'model-{i}'.encode()) for i, r in enumerate(recipients)]
    outer, manifest, signature = seal_bundle(gate, admission, recipients, entries, key, issuer)
    issued = create_issuance(pre, registry.public_key(), manifest, signature, issuer.public_key(),
                             gate, admission, recipients, registry)
    checkpoint = create_checkpoint(pre, issued, registry.public_key(), witness, 'c' * 64,
                                   prior, 'b' * 64, now=1010)
    context = request_context(public, keyctx, sha(outer), manifest, 'd' * 64)
    proofs = [prove(public, share, context) for share in shares]
    _, receipt = reconstruct(public, shares, proofs, context, issuer)

    def pubhex(private):
        return private.public_key().public_bytes(serialization.Encoding.Raw,
                                                  serialization.PublicFormat.Raw).hex()

    anchors = {'issuer': pubhex(issuer), 'registry': pubhex(registry), 'witness': pubhex(witness)}
    reg.update(replica_ids=[f'C-E0-U{i}' for i in (1, 2, 3)], words=[word.tolist() for word in words],
               p=tardos.p.tolist(), threshold=tardos.threshold)
    for name, data in {
        'anchors.json': anchors, 'registration_before_training.json': reg,
        'registry_precommit.json': pre, 'registry_precommit_checkpoint.json': prior,
        'registry_issued.json': issued, 'registry_checkpoint.json': checkpoint,
        'fingerprint_openings.json': openings, 'reconstruction_receipt.json': receipt,
        'signed_manifest.json': {'manifest': manifest, 'signature_hex': signature.hex(),
                                 'issuer_public_key': anchors['issuer']},
        'model_metrics.json': {'outer_cipher_sha256': sha(outer)},
    }.items():
        dump(tmp_path / name, data)
    (tmp_path / 'outer_bundle.aesgcm').write_bytes(outer)
    return tmp_path, anchors, reg, outer


def test_local_audit_checks_real_ciphertext_and_signed_parameters(audit_run):
    root, anchors, reg, outer = audit_run
    result = audit_directory(root, root / 'anchors.json', 'c' * 64, now=1011)
    assert result['outer_ciphertext_hash_rechecked']
    assert result['checkpoint_freshness'] == 'valid_at_local_clock'

    changed = copy.deepcopy(reg)
    changed['threshold'] += 1
    dump(root / 'registration_before_training.json', changed)
    with pytest.raises(ValueError):
        audit_directory(root, root / 'anchors.json', 'c' * 64, now=1011)
    dump(root / 'registration_before_training.json', reg)

    (root / 'outer_bundle.aesgcm').write_bytes(outer + b'x')
    with pytest.raises(ValueError):
        audit_directory(root, root / 'anchors.json', 'c' * 64, now=1011)
    (root / 'outer_bundle.aesgcm').write_bytes(outer)
    anchors['issuer'] = Ed25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    dump(root / 'anchors.json', anchors)
    with pytest.raises(ValueError):
        audit_directory(root, root / 'anchors.json', 'c' * 64, now=1011)


def test_expired_checkpoint_and_historical_mode(audit_run):
    root, _, _, _ = audit_run
    with pytest.raises(ValueError):
        audit_directory(root, root / 'anchors.json', 'c' * 64, now=1310)
    result = audit_directory(root, root / 'anchors.json', 'c' * 64, historical=True)
    assert result['checkpoint_freshness'] == 'not_established_historical_replay'


def test_recomputes_trace_results_from_all_saved_responses(audit_run):
    root, _, reg, _ = audit_run
    tardos = Tardos(n=3)
    for rid, word in zip(reg['replica_ids'], reg['words']):
        scores = [1 if bit else -1 for bit in word for _ in range(3)]
        decoded = [reg['replica_ids'][i] for i in tardos.decode(reg['words'], word)]
        dump(root / f'{rid}_batch_http.json', {
            'response': {'score_differences': scores}, 'result': {'decoded_replicas': decoded},
        })
    result = audit_directory(root, root / 'anchors.json', 'c' * 64, now=1011)
    assert [row['matched_bits'] for row in result['trace_transcript_consistency']] == [256] * 3
    changed = json.loads((root / 'C-E0-U1_batch_http.json').read_text())
    changed['result']['decoded_replicas'] = ['wrong-replica']
    dump(root / 'C-E0-U1_batch_http.json', changed)
    with pytest.raises(ValueError):
        audit_directory(root, root / 'anchors.json', 'c' * 64, now=1011)
