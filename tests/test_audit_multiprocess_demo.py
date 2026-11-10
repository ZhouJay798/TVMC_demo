import copy
import io
import json
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from audit_multiprocess_demo import audit_directory
from run_personal_delivery import setup
from tvmc.bundle_delivery import make_inner, seal_bundle
from tvmc.crypto import prove
from tvmc.personal_delivery import replica_seed, sha
from tvmc.reconstruction import reconstruct, request_context
from tvmc.tardos import Tardos
from tvmc.trusted_registry import create_issuance, create_precommit


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def pubhex(private):
    return private.public_key().public_bytes(serialization.Encoding.Raw,
                                              serialization.PublicFormat.Raw).hex()


def zip_payload(index):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('base/config.json', '{}')
        z.writestr('base/model.safetensors', f'fixture-{index}'.encode())
    return stream.getvalue()


@pytest.fixture
def audit_run(tmp_path):
    root = tmp_path/'run'
    root.mkdir()
    gate, admission, (key, public, shares, _, keyctx), _, recipients, registration = setup('a'*64)
    issuer, registrar = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    anchors = {'issuer': pubhex(issuer), 'registrar': pubhex(registrar)}
    anchors_path = tmp_path/'trusted-anchors.json'
    dump(anchors_path, anchors)
    tardos = Tardos(n=3)
    words = [tardos.encode(replica_seed(admission, record)).tolist() for record in recipients]
    ids = [f'{admission.sid}-U{i}' for i in admission.participants]
    pre, openings = create_precommit(gate, admission, recipients, words, registrar, tardos)
    plaintexts = [zip_payload(i) for i in range(1,4)]
    entries = [make_inner(gate, admission, record, payload)
               for record, payload in zip(recipients, plaintexts)]
    outer, manifest, signature = seal_bundle(gate, admission, recipients, entries, key, issuer)
    issued = create_issuance(pre, registrar.public_key(), manifest, signature, issuer.public_key(),
                             gate, admission, recipients, registrar)
    context = request_context(public, keyctx, sha(outer), manifest, 'b'*64)
    _, receipt = reconstruct(public, shares, [prove(public, share, context) for share in shares], context, issuer)
    registration.update(replica_ids=ids, issuer_public=anchors['issuer'], registrar_public=anchors['registrar'])
    checks = {'distinct_processes': True, 'all_models_trace_to_their_copy': True}
    for name, obj in {
        'public_registration.json': registration,
        'signed_manifest.json': {'manifest': manifest, 'signature': signature.hex(),
                                 'issuer_public': anchors['issuer'], 'registrar_public': anchors['registrar']},
        'registry_precommit.json': pre, 'registry_issued.json': issued,
        'audit_codebook.json': {'replica_ids': ids, 'words': words, 'openings': openings,
                                'p': tardos.p.tolist(), 'threshold': tardos.threshold},
        'reconstruction_receipt.json': receipt,
        'metrics.json': {'status': 'completed', 'checks': checks, 'base_sha256': admission.base_sha256},
    }.items():
        dump(root/name, obj)
    (root/'outer_bundle.aesgcm').write_bytes(outer)
    for i, (payload, entry) in enumerate(zip(plaintexts, entries), 1):
        target = root/'custodians'/f'U{i}'/'received_model.zip'
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        for suffix, value in [('cipher', entry['ciphertext']), ('key', entry['wrapped_key'])]:
            target = root/'encrypted_transit'/f'U{i}.{suffix}'
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(value)
    return root, anchors_path, anchors


def test_success_rechecks_real_payloads_without_exposing_secrets(audit_run):
    root, anchor, _ = audit_run
    result = audit_directory(root, anchor)
    assert result['audit_status'] == 'verified_saved_records'
    assert len(result['recipient_payloads']) == 3
    assert result['run_reported_status'] == 'completed'
    assert result['run_reported_status_authenticated'] is False
    assert result['outer_ciphertext_decrypted_or_contents_verified'] is False
    assert result['recipient_file_decryption_origin_verified'] is False
    assert result['fingerprint_actually_embedded_in_model_verified'] is False
    assert result['checkpoint_freshness_and_rollback_protection'] == 'not_available'
    assert result['request_nonce_freshness_and_replay_protection'] == 'not_verified'
    assert result['trace_and_model_provenance'] == 'not_verified_no_raw_trace_or_remote_attestation'
    rendered = json.dumps(result)
    codebook = json.loads((root/'audit_codebook.json').read_text())
    for opening in codebook['openings'].values():
        assert opening not in rendered


def test_rejects_wrong_anchors_and_tampered_signed_registry(audit_run):
    root, anchor, keys = audit_run
    changed = dict(keys, issuer=pubhex(Ed25519PrivateKey.generate()))
    dump(anchor, changed)
    with pytest.raises(ValueError):
        audit_directory(root, anchor)
    dump(anchor, keys)
    pre_path = root/'registry_precommit.json'
    pre = json.loads(pre_path.read_text())
    pre['body']['entries'][0]['recipient'] = 2
    dump(pre_path, pre)
    with pytest.raises(ValueError):
        audit_directory(root, anchor)


@pytest.mark.parametrize('relative', ['outer_bundle.aesgcm', 'custodians/U2/received_model.zip',
                                      'encrypted_transit/U3.cipher', 'encrypted_transit/U1.key'])
def test_rejects_modified_actual_files(audit_run, relative):
    root, anchor, _ = audit_run
    path = root/relative
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ValueError):
        audit_directory(root, anchor)


def test_rejects_wrong_codeword_and_opening(audit_run):
    root, anchor, _ = audit_run
    path = root/'audit_codebook.json'
    original = json.loads(path.read_text())
    changed = copy.deepcopy(original)
    changed['words'][0][0] ^= 1
    dump(path, changed)
    with pytest.raises(ValueError):
        audit_directory(root, anchor)
    changed = copy.deepcopy(original)
    changed['openings'][changed['replica_ids'][0]] = 'a'*64
    dump(path, changed)
    with pytest.raises(ValueError):
        audit_directory(root, anchor)


@pytest.mark.parametrize('target_kind', ['file', 'directory'])
def test_rejects_symlinked_custodian_paths(audit_run, target_kind):
    root, anchor, _ = audit_run
    directory = root/'custodians'/'U1'
    path = directory/'received_model.zip' if target_kind == 'file' else directory
    renamed = path.with_name(path.name+'.real')
    path.rename(renamed)
    path.symlink_to(renamed.name, target_is_directory=target_kind == 'directory')
    with pytest.raises((OSError, ValueError)):
        audit_directory(root, anchor)


def test_self_reported_failed_checks_produce_exit_one(audit_run):
    root, anchor, _ = audit_run
    metrics_path = root/'metrics.json'
    metrics = json.loads(metrics_path.read_text())
    metrics.update(status='completed_with_failed_checks', checks={'model_trace': False})
    # Path strings inside metrics must never be opened as recipient payloads.
    metrics['recipients'] = [{'payload_path': '/etc/passwd'}]
    dump(metrics_path, metrics)
    result = audit_directory(root, anchor)
    assert result['audit_status'] == 'verified_saved_records'
    assert result['run_reported_status'] == 'completed_with_failed_checks'
    executable = Path(__file__).resolve().parents[1]/'audit_multiprocess_demo.py'
    proc = subprocess.run([sys.executable, str(executable), '--run-dir', str(root),
                           '--trusted-anchors', str(anchor)], capture_output=True, text=True)
    assert proc.returncode == 1
    assert json.loads(proc.stdout)['run_reported_status'] == 'completed_with_failed_checks'


@pytest.mark.parametrize('file,mutate', [
    ('signed_manifest.json', lambda value: value['manifest']['entries'][0]['context'].__setitem__('recipient', 2)),
    ('reconstruction_receipt.json', lambda value: value['record'].__setitem__('participants', [1, 2])),
    ('public_registration.json', lambda value: value['admission'].__setitem__('seed', 'a'*64)),
])
def test_rejects_tampered_manifest_receipt_and_admission(audit_run, file, mutate):
    root, anchor, _ = audit_run
    path = root/file
    data = json.loads(path.read_text())
    mutate(data)
    dump(path, data)
    with pytest.raises((ValueError, KeyError, InvalidSignature)):
        audit_directory(root, anchor)


def test_cli_error_ends_with_a_real_newline(audit_run):
    root, anchor, keys = audit_run
    dump(anchor, dict(keys, registrar=pubhex(Ed25519PrivateKey.generate())))
    executable = Path(__file__).resolve().parents[1]/'audit_multiprocess_demo.py'
    proc = subprocess.run([sys.executable, str(executable), '--run-dir', str(root),
                           '--trusted-anchors', str(anchor)], capture_output=True)
    assert proc.returncode == 2 and proc.stderr.endswith(b'\n')
    assert not proc.stderr.endswith(b'\\n')
