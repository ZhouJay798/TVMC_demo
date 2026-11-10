import os
import stat
import zipfile
from pathlib import Path

import pytest

from tvmc.offline_fixture import make_fixture
from tvmc.process_payloads import (_extract_model_archive, build_payloads,
                                   detect_payload, file_sha256)
from tvmc.tardos import Tardos


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp('process_payloads')
    source = root / 'source'
    data = make_fixture(source)
    before = file_sha256(source / 'model.safetensors')
    tardos = Tardos(n=3)
    words = [tardos.encode(f'fixture-recipient-{i}').tolist() for i in range(1, 4)]
    ids = ['C-E0-U1', 'C-E0-U2', 'C-E0-U3']
    result = build_payloads(output_dir=root/'payloads', words=words, replica_ids=ids,
                            offline_fixture=True, model_path=source, fixture_data=data, steps=1, threads=1)
    assert file_sha256(source/'model.safetensors') == before == result['base_sha256']
    return root, words, ids, result


def test_fixture_builds_three_real_merged_archives_without_secrets(built):
    root, words, ids, result = built
    assert result['scope'] == 'synthetic_fixture'
    assert result['training_count'] == 3 and len(result['prompts']) == 768
    for i, row in enumerate(result['replicas']):
        path = Path(row['path'])
        assert path.is_absolute() and path.is_file() and row['replica_id'] == ids[i]
        assert file_sha256(path) == row['sha256']
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            assert 'base/model.safetensors' in names and 'base/config.json' in names
            assert 'delivery_format.json' in names and 'adapter.pt' not in names
            assert all(name == 'delivery_format.json' or name.startswith('base/') for name in names)
            assert not any('registration' in name or 'seed' in name or 'word' in name for name in names)


def test_detector_runs_actual_reloaded_fixture_model(built):
    root, words, ids, result = built
    decoded = detect_payload(zip_path=result['replicas'][0]['path'], prompts=result['prompts'],
                             words=words, replica_ids=ids, cache=root/'cache', threads=1)
    assert decoded['observed_bits'] == 256
    assert len(decoded['scores']) == len(decoded['correct_bits']) == 3
    assert set(decoded['decoded_replicas']) <= set(ids)
    assert decoded['individual_leaker_proven'] is False
    # A single training step is deliberately not a quality or identification claim.


def test_rejects_archive_path_traversal_and_duplicate_entries(tmp_path):
    target = tmp_path/'extract'; target.mkdir()
    path = tmp_path/'bad.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('../escape', b'bad')
    with pytest.raises(ValueError):
        _extract_model_archive(path, target)
    assert not (tmp_path/'escape').exists()
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('base/config.json', '{}')
        with pytest.warns(UserWarning, match='Duplicate name'):
            z.writestr('base/config.json', '{}')
    with pytest.raises(ValueError):
        _extract_model_archive(path, target)


def test_rejects_ambiguous_codebook_and_insecure_output(tmp_path):
    words = [[0] * 256] * 3
    with pytest.raises(ValueError):
        build_payloads(output_dir=tmp_path/'a', words=words, replica_ids=['1', '2', '3'],
                       offline_fixture=True, steps=1)
    t = Tardos(n=3)
    words = [t.encode(f'distinct{i}').tolist() for i in range(3)]
    out = tmp_path/'public'; out.mkdir(mode=0o755); os.chmod(out, 0o755)
    with pytest.raises(ValueError):
        build_payloads(output_dir=out, words=words, replica_ids=['1', '2', '3'],
                       offline_fixture=True, steps=1)
