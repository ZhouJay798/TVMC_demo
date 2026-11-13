"""Compare project row/group INT8 float simulation on one delivered U1 copy.

This uses the registered embedding probes and the existing compressed() code.
Results are local sensitive research records, not a population-level guarantee.
"""
import argparse
import gc
import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run_finetune_fingerprint_experiment import (
    _extract_model_archive, corpus_windows, detect, dump, load_local,
    perplexity, verify_source,
)
from tvmc.robustness import compressed, group_quantize


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@contextmanager
def quantized_head(model, method):
    weight = model.lm_head.weight
    original = weight.detach().clone()
    try:
        with torch.no_grad():
            if method == 'group_int8':
                transformed = group_quantize(weight)
            elif method == 'row_int8':
                scale = weight.abs().amax(1, keepdim=True).clamp_min(1e-8)/127
                transformed = (weight/scale).round().clamp(-127, 127)*scale
            else:
                raise ValueError(method)
            weight.copy_(transformed)
        yield
    finally:
        with torch.no_grad():
            weight.copy_(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, default=Path('work/multiprocess_real/run2'))
    parser.add_argument('--corpus', type=Path, default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/huggingface'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--with-head', action='store_true', help='also quantize lm_head; tied embeddings will change too')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = verify_source(args.run_dir.resolve(), 1)
    windows, corpus_hash = corpus_windows(args.corpus)
    if corpus_hash != source['source_metrics'].get('text_corpus_sha256'):
        raise ValueError('corpus mismatch')
    result = {'status': 'running', 'source': 'U1 already-embedded delivered model',
        'source_zip_sha256': source['payload_sha256'], 'corpus_sha256': corpus_hash,
        'scope': ('all transformer nn.Linear and lm_head weights; embeddings change only if tied to head' if args.with_head else
                  'all transformer nn.Linear weights; embeddings, layernorm, lm_head excluded'),
        'group_size': 32, 'method': 'symmetric max-abs weight-only quantize/dequantize, range [-127,127], float32 execution',
        'evaluation': 'same 768 probes used in embedding; PPL 16 held-out TinyShakespeare windows 384-399, 64 tokens max, same corpus',
        'trust': 'signatures verified using public keys from same saved run; not independent authentication',
        'branches': {}}
    dump(out/'metrics.json',result)
    with tempfile.TemporaryDirectory(prefix='quant-u1-', dir=out) as temp:
        model_dir = _extract_model_archive(source['payload'], temp)
        model, tokenizer, token_ids = load_local(model_dir, args.cache, args.threads)
        head_hash = tensor_hash(model.lm_head.weight)
        embedding_hash = tensor_hash(model.model.embed_tokens.weight)
        tied = model.lm_head.weight.untyped_storage().data_ptr() == model.model.embed_tokens.weight.untyped_storage().data_ptr()
        result['head_embedding_shared_storage'] = tied
        print('HEAD_EMBEDDING_SHARED_STORAGE', tied, flush=True)
        eval_texts = windows[384:400]
        baseline_bits = {}
        for mode in ('baseline', 'group_int8', 'row_int8'):
            print('BRANCH', mode, flush=True)
            def measure(info):
                trace = detect(model, tokenizer, token_ids, source['prompts'], source['words'], source['ids'], source['tardos'])
                quality = perplexity(model, tokenizer, eval_texts, 64)
                head_changed = head_hash != tensor_hash(model.lm_head.weight)
                embedding_changed = embedding_hash != tensor_hash(model.model.embed_tokens.weight)
                if not args.with_head and (head_changed or embedding_changed):
                    raise AssertionError('lm_head or embeddings changed unexpectedly')
                if mode != 'baseline' and args.with_head and not head_changed:
                    raise AssertionError('head unexpectedly did not change')
                if mode != 'baseline' and args.with_head and embedding_changed != tied:
                    raise AssertionError('embedding storage changed inconsistently with head tying')
                modes = {}
                for name, part in [('historical', trace), ('actual_last', trace['correct_last_token'])]:
                    bits = np.asarray(part['observed_bits'])
                    if mode == 'baseline': baseline_bits[name] = bits
                    modes[name] = {'matched_per_recipient': part['correct_bits_per_replica'],
                        'scores': part['tardos_scores'], 'threshold': part['threshold'],
                        'decoded_replicas': part['decoded_replicas'],
                        'bits_changed_from_baseline': int((bits != baseline_bits[name]).sum())}
                summary = {'compression': info, 'readout': modes, 'quality': quality,
                    'output_head_changed': head_changed, 'embedding_changed': embedding_changed}
                if mode != 'baseline':
                    base = result['branches']['baseline']['quality']['perplexity']
                    summary['ppl_relative_change_pct'] = 100*(quality['perplexity']/base-1)
                return summary
            if mode == 'baseline':
                summary = measure(None)
            else:
                with compressed(SimpleNamespace(model=model), mode) as info:
                    if args.with_head:
                        with quantized_head(model, mode):
                            summary = measure(dict(info, extra='lm_head, plus embeddings if tied'))
                    else:
                        summary = measure(info)
            if head_hash != tensor_hash(model.lm_head.weight) or embedding_hash != tensor_hash(model.model.embed_tokens.weight):
                raise AssertionError('head or embeddings failed to restore')
            result['branches'][mode] = summary
            dump(out/'metrics.json',result)
            print('RESULT', json.dumps({'mode':mode,'actual_last':summary['readout']['actual_last'],
                  'historical':summary['readout']['historical'],'quality':summary['quality'],
                  'relative_pct': summary.get('ppl_relative_change_pct')},ensure_ascii=False),flush=True)
        del model, tokenizer
        gc.collect()
    result['status']='completed'
    dump(out/'metrics.json',result)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
