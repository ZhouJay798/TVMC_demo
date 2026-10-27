"""Run from repository root: .venv/bin/python results/compression_wikitext2_20261027/rerun.py"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from tvmc.model import ModelLab
from tvmc.robustness import corpus, utility, compressed, DATA_REVISION

cache = ROOT / '.cache/huggingface'
source = ROOT / 'outputs/20261026T173212Z-0a7b2d'
research = ROOT / 'outputs/robustness/20261026T173325Z-5005fd'
data = corpus(cache, offline=True, test_offset=128)
lab = ModelLab(str(source / 'decrypted_model'), cache, 'cpu', 4, True)
states = {'clean': lab.snapshot()}
paths = {'original_fingerprint': source / 'user_1_lora.pt',
         'natural_redundant': research / 'natural_redundant_user1.pt'}
for profile, path in paths.items():
    states[profile] = torch.load(path, map_location='cpu', weights_only=True)
result = {'dataset': 'Salesforce/wikitext wikitext-2-raw-v1',
          'revision': DATA_REVISION, 'test_offset': 128, 'documents': 64,
          'max_tokens_per_document': 96,
          'test_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in data['test']],
          'checkpoint_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
          'note': 'Existing user 1 checkpoints; fresh paired evaluation, no retraining. Not the full WikiText-2 benchmark.',
          'measurements': {}}
for method in ('uncompressed', 'row_int8', 'magnitude30'):
    from contextlib import nullcontext
    with (nullcontext({}) if method == 'uncompressed' else compressed(lab, method)) as info:
        rows = {}
        for profile, state in states.items():
            lab.restore(state)
            measured, _ = utility(lab, data['test'])
            measured.pop('next_token_agreement')  # No reference predictions in this PPL-only run.
            rows[profile] = measured
            print(method, profile, measured, flush=True)
        result['measurements'][method] = {'parameters': info, 'models': rows}
base = result['measurements']['uncompressed']['models']['clean']['perplexity']
for method, entry in result['measurements'].items():
    for profile, row in entry['models'].items():
        own = result['measurements']['uncompressed']['models'][profile]['perplexity']
        row['increase_vs_clean_original'] = row['perplexity'] / base - 1
        row['increase_vs_own_uncompressed'] = row['perplexity'] / own - 1
Path(__file__).with_name('metrics.json').write_text(json.dumps(result, indent=2)+'\n')
