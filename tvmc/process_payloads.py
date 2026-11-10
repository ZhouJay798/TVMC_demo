"""Create actual merged-model archives for the local multi-process demo.

Models and trigger prompts in this module are a research fixture, not a secure
multi-host delivery system.  A random offline fixture is never a pretrained
model. The returned prompts and codewords are sensitive research material and
must not be exposed in a production evidence workflow.
"""
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .merged_delivery import save_merged_model
from .model import ModelLab
from .offline_fixture import make_fixture
from .plaintext_corpus import load_plaintext_corpus
from .robustness import aggregate, corpus, embed_natural, margins, natural_triggers, training_cache
from .tardos import Tardos


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _fixture_data():
    """Reconstruct the fixture's public text without rewriting model files."""
    triggers = [' '.join([f'doc{i:03d}'] * 32) for i in range(256)]
    neutral = ' '.join(['neutral', 'context'] * 32)
    return {'triggers': triggers, 'calibration': [neutral] * 8, 'test': [neutral] * 8}


def _validate_codebook(words, replica_ids):
    if (not isinstance(replica_ids, (list, tuple)) or len(replica_ids) != 3 or
        not all(isinstance(item, str) and item for item in replica_ids) or
        len(set(replica_ids)) != 3):
        raise ValueError('three distinct replica IDs required')
    try:
        # Do not coerce arbitrary floats into bits: this is an evidence input.
        arr = np.asarray(words)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid codebook') from exc
    if arr.shape != (3, 256) or not np.isin(arr, [0, 1]).all():
        raise ValueError('three binary 256-bit codewords required')
    if len({tuple(row.tolist()) for row in arr}) != 3:
        raise ValueError('duplicate replica codewords')
    return arr.astype(np.int8)


def _archive_merged(lab, output_path, temporary_parent):
    """Write only model/tokenizer files and a format marker to a new zip."""
    with tempfile.TemporaryDirectory(prefix='.model-', dir=temporary_parent) as temp:
        save_merged_model(lab, temp)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(output_path, flags, 0o600)
        try:
            with os.fdopen(fd, 'wb') as stream, zipfile.ZipFile(stream, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
                for path in sorted(Path(temp).iterdir()):
                    if path.is_symlink() or not path.is_file():
                        raise ValueError('unexpected merged model file')
                    archive.write(path, f'base/{path.name}')
                archive.writestr('delivery_format.json', json.dumps({'format': 'merged_model/v1'}))
        except BaseException:
            # Only a newly created file is removed, never a pre-existing file.
            output_path.unlink(missing_ok=True)
            raise


def build_payloads(*, output_dir, words, replica_ids, offline_fixture=False,
                   model_path=None, fixture_data=None, text_corpus=None, cache=None,
                   steps=1200, threads=2):
    """Train three distinct registered codewords into real model weights.

    Returns local archive paths/hashes plus prompts for local test observation.
    The offline fixture uses a deterministic *random* small Transformer and
    synthetic text. If model_path is provided in fixture mode, it is read only;
    the caller is responsible for supplying a matching fixture source/data.
    """
    codebook = _validate_codebook(words, replica_ids)
    if type(steps) is not int or steps < 1 or type(threads) is not int or threads < 1:
        raise ValueError('steps and threads must be positive integers')
    if offline_fixture and text_corpus is not None:
        raise ValueError('offline fixture cannot use an external text corpus')
    if not offline_fixture and (fixture_data is not None or model_path is None):
        raise ValueError('local model path required; fixture_data is fixture-only')
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not out.is_dir() or stat.S_IMODE(out.stat().st_mode) & 0o077:
        raise ValueError('output directory must not be accessible to group or others')
    cache = Path(cache).resolve() if cache is not None else out / '.hf-cache'
    fixture_temp = None
    try:
        if offline_fixture and model_path is None:
            fixture_temp = tempfile.TemporaryDirectory(prefix='tvmc-process-source-')
            source = Path(fixture_temp.name) / 'model'
            generated_data = make_fixture(source)
            data = fixture_data if fixture_data is not None else generated_data
        else:
            source = Path(model_path).resolve()
            data = ((fixture_data if fixture_data is not None else _fixture_data()) if offline_fixture else
                    (load_plaintext_corpus(text_corpus) if text_corpus is not None else corpus(cache, True, 256)))
        weight_path = source / 'model.safetensors'
        if not weight_path.is_file():
            raise FileNotFoundError(f'model.safetensors not found in {source}')
        if len(data.get('triggers', [])) != 256 or not data.get('calibration'):
            raise ValueError('256 trigger documents and calibration text are required')
        base_sha256 = file_sha256(weight_path)
        lab = ModelLab(str(source), cache, 'cpu', threads, True)
        prompts = natural_triggers(lab, data['triggers'], 3)
        if len(prompts) != 768:
            raise ValueError('expected exactly 768 trigger prompts')
        features = lab.features(prompts)
        calibration = training_cache(lab, data['calibration'])
        replicas = []
        for index, (replica_id, word) in enumerate(zip(replica_ids, codebook)):
            torch.manual_seed(10101 + index)
            training = embed_natural(lab, features, np.repeat(word, 3), calibration, steps=steps)
            path = out / f'replica_{index + 1}.zip'
            _archive_merged(lab, path, out)
            replicas.append({'replica_id': replica_id, 'path': str(path),
                             'sha256': file_sha256(path), 'training': training})
        return {'scope': 'synthetic_fixture' if offline_fixture else 'local_model',
                'base_sha256': base_sha256, 'prompts': prompts, 'replicas': replicas,
                'training_count': len(replicas), 'steps': steps}
    finally:
        if fixture_temp is not None:
            fixture_temp.cleanup()


def _extract_model_archive(zip_path, target, max_uncompressed_bytes=4 * 1024**3):
    """Extract only regular, flat base files; reject traversals and duplicates."""
    target = Path(target)
    seen, total, marker = set(), 0, False
    with zipfile.ZipFile(zip_path, 'r') as archive:
        infos = archive.infolist()
        if len(infos) > 256:
            raise ValueError('too many model archive members')
        for info in infos:
            name = info.filename
            path = PurePosixPath(name)
            mode = info.external_attr >> 16
            if ('\\' in name or path.is_absolute() or '..' in path.parts or
                name in seen or info.is_dir() or stat.S_ISLNK(mode)):
                raise ValueError('unsafe model archive member')
            seen.add(name)
            if name == 'delivery_format.json':
                if info.file_size > 4096:
                    raise ValueError('oversized delivery format marker')
                if json.loads(archive.read(info)) != {'format': 'merged_model/v1'}:
                    raise ValueError('unexpected delivery format')
                marker = True
                continue
            if len(path.parts) != 2 or path.parts[0] != 'base' or path.parts[1] in ('', '.', '..'):
                raise ValueError('unexpected model archive path')
            total += info.file_size
            if total > max_uncompressed_bytes:
                raise ValueError('model archive too large')
            base = target / 'base'
            base.mkdir(exist_ok=True, mode=0o700)
            with archive.open(info) as source, (base / path.parts[1]).open('xb') as dest:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
    if not marker or 'base/model.safetensors' not in seen or 'base/config.json' not in seen:
        raise ValueError('incomplete merged model archive')
    return target / 'base'


def detect_payload(*, zip_path, prompts, words, replica_ids, cache=None, threads=2):
    """Run the actual extracted model and decode scores, without an HTTP call."""
    codebook = _validate_codebook(words, replica_ids)
    if type(threads) is not int or threads < 1:
        raise ValueError('threads must be a positive integer')
    if not isinstance(prompts, list) or len(prompts) != 768 or not all(isinstance(p, str) for p in prompts):
        raise ValueError('exactly 768 text prompts required')
    with tempfile.TemporaryDirectory(prefix='tvmc-process-detect-') as temp:
        base = _extract_model_archive(zip_path, temp)
        cache_dir = Path(cache).resolve() if cache is not None else Path(temp) / '.hf-cache'
        lab = ModelLab(str(base), cache_dir, 'cpu', threads, True)
        scores_by_prompt = margins(lab, prompts)
        if len(scores_by_prompt) != 768 or not np.isfinite(scores_by_prompt).all():
            raise ValueError('invalid model output')
        observed = aggregate(scores_by_prompt, 3)
        tardos = Tardos(n=len(replica_ids))
        scores = tardos.scores(codebook, observed)
        decoded = tardos.decode(codebook, observed)
        return {'observed_bits': int((observed >= 0).sum()),
                'correct_bits': [int((observed == word).sum()) for word in codebook],
                'scores': scores.tolist(), 'threshold': tardos.threshold,
                'decoded_replicas': [replica_ids[i] for i in decoded],
                'query_scope': 'local full model inference; no HTTP or remote provenance',
                'individual_leaker_proven': False}
