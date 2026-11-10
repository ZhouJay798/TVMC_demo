import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_cli_runs_real_process_roles_with_separate_model_archives(tmp_path):
    """One-step training is a smoke test, not a fingerprint accuracy test."""
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / 'run'
    env = dict(os.environ, HF_HOME=str(tmp_path / 'hf-cache'))
    proc = subprocess.run([sys.executable, 'run_multiprocess_demo.py', '--offline-fixture',
                           '--steps', '1', '--output', str(out)], cwd=root, env=env,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode in (0, 2), proc.stdout + '\n' + proc.stderr
    metrics = json.loads((out / 'metrics.json').read_text())
    assert metrics['model_scope'] == 'synthetic_fixture'
    assert metrics['status'] == ('completed' if all(metrics['checks'].values()) else 'completed_with_failed_checks')
    assert len(set(metrics['processes'].values())) == 8
    assert metrics['checks']['all_three_recover_session_secret']
    assert metrics['checks']['all_six_wrong_recipient_packages_rejected']
    assert metrics['checks']['all_owner_model_hashes_match']
    assert metrics['checks']['invalid_issuer_signature_rejected']
    assert metrics['checks']['tampered_registry_rejected']
    assert (out / 'public_registration.json').is_file()
    assert (out / 'MULTIPROCESS_REPORT.md').is_file()
    assert not list(out.glob('.issuer_plain_*'))
    for i in (1, 2, 3):
        with zipfile.ZipFile(out / 'custodians' / f'U{i}' / 'received_model.zip') as z:
            names = set(z.namelist())
        assert 'base/model.safetensors' in names
        assert 'adapter.pt' not in names and 'registration.json' not in names
