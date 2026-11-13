"""Small paired robustness study of locally registered recipient models.

This is a research harness, not a security guarantee.  Its audit anchors come
from the same local run.  It deliberately retains both historical and actual
last-token readouts because the delivered copies used the historical index.
The corpus is not a business-task benchmark.  Outputs contain sensitive
derived evidence and should not be published without review.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from run_finetune_fingerprint_experiment import (
    _extract_model_archive, corpus_windows, detection_positions, dump,
    load_local, train, verify_source,
)
from tvmc.robustness import aggregate, group_quantize


def pack_row_int8(model, with_head, output):
    """Actual packed PyTorch per-output-channel qint8 dynamic linear layers.

    Real integer weights and per-row scales are serialized; the dynamic CPU
    operator additionally quantizes activations at inference. Embeddings are
    excluded.  The output head is optionally included.
    """
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
    from torch.ao.quantization import per_channel_dynamic_qconfig
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and
               (name.startswith('model.layers.') or (with_head and name == 'lm_head'))]
    model.eval()
    for name, module in modules:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        module.qconfig = per_channel_dynamic_qconfig
        setattr(parent, child, DynamicLinear.from_float(module))
    model.config.use_cache = False
    packed = {name: {'weight': model.get_submodule(name).weight(),
                     'bias': model.get_submodule(name).bias()} for name, _ in modules}
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    torch.save({'format': 'pytorch_dynamic_per_channel_qint8', 'modules': packed}, output)
    output.chmod(0o600)
    check = torch.load(output, map_location='cpu', weights_only=True)
    if len(check['modules']) != len(modules) or any(
        w['weight'].dtype != torch.qint8 or w['weight'].qscheme() != torch.per_channel_affine
        for w in check['modules'].values()
    ):
        raise ValueError('packed qint8 round trip failed')
    del packed, check
    return {'format': 'PyTorch dynamic per-channel qint8', 'modules': len(modules),
            'head_quantized': with_head, 'embedding_quantized': False,
            'integer_weights_serialized': True, 'pytorch_cpu_quantized_operator': True,
            'packed_weights_bytes': output.stat().st_size}


def apply_row_simulation(model):
    """Per-row INT8 roundtrip for sequential full-tuning -> INT8 study."""
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and (name.startswith('model.layers.') or name == 'lm_head'):
                w = module.weight
                scale = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
                w.copy_((w / scale).round().clamp(-127, 127) * scale)
    return {'method': 'row INT8 roundtrip simulation, transformer and output head',
            'integer_storage': False, 'execution': 'float32'}


def prune_magnitude(model, fraction):
    zeros = total = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not name.startswith('model.layers.') or not isinstance(module, nn.Linear):
                continue
            weight = module.weight
            count = weight.numel()
            k = max(1, int(count * fraction))
            cut = weight.abs().reshape(-1).kthvalue(k).values
            weight.mul_(weight.abs() > cut)
            total += count
            zeros += int((weight == 0).sum())
    return {'scope': 'transformer linear weights, per-matrix magnitude pruning',
            'requested_fraction': fraction, 'actual_zero_fraction': zeros/total,
            'weights': total, 'output_head_pruned': False}


def prune_structured_mlp(model, fraction):
    """Zero entire coupled MLP neurons (same shapes, no runtime acceleration)."""
    n = total = 0
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            hidden = mlp.gate_proj.weight.shape[0]
            count = max(1, int(hidden * fraction))
            importance = (mlp.gate_proj.weight.square().sum(1) +
                          mlp.up_proj.weight.square().sum(1) +
                          mlp.down_proj.weight.square().sum(0))
            selected = torch.topk(importance, count, largest=False).indices
            mlp.gate_proj.weight[selected] = 0
            mlp.up_proj.weight[selected] = 0
            mlp.down_proj.weight[:, selected] = 0
            n += count
            total += hidden
    return {'method': 'structured MLP neuron zeroing, shapes unchanged',
            'neurons_zeroed': n, 'total_neurons': total,
            'actual_neuron_fraction': n/total, 'output_head_pruned': False}


def erase_known_probes(model, tokenizer, prompts, word, token_ids):
    """Strong probe-aware head update; not a reproduction of MEraser.

    The attacker knows one of three prefixes for each bit and its codeword.
    Optimizes only 0/1 head rows to reverse their margins. Other two prefixes
    are examined separately, but are correlated with the exposed prefixes.
    """
    model.eval()
    features = []
    with torch.no_grad():
        exposed = prompts[::3]
        for offset in range(0, len(exposed), 16):
            batch = tokenizer(exposed[offset:offset+16], padding=True, truncation=True,
                              max_length=40, return_tensors='pt', return_token_type_ids=False)
            hidden = model.model(**batch, use_cache=False).last_hidden_state
            last = detection_positions(batch['attention_mask'])[1]
            features.append(hidden[torch.arange(len(last)), last].detach())
    h = torch.cat(features)
    # Bias is absent for the project model; fail rather than silently ignore it.
    if model.lm_head.bias is not None:
        raise ValueError('probe-aware attack expects bias-free lm_head')
    original = model.lm_head.weight[token_ids].detach().clone()
    rows = nn.Parameter(original.clone())
    labels = 1 - torch.as_tensor(word, dtype=torch.long)
    opt = torch.optim.Adam([rows], lr=0.002)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        pair = F.linear(h, rows)
        loss = F.cross_entropy(pair, labels) + 0.01 * (rows-original).square().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.lm_head.weight[token_ids] = rows
    return {'method': 'probe-aware reverse-label output-head optimization',
            'assumptions': 'attacker knows codeword and one prefix per bit; 256 exposed probes',
            'training_probes': len(exposed), 'untouched_prefixes_per_bit': 2,
            'untouched_prefixes_are_independent': False, 'steps': 80,
            'learning_rate': .002, 'trainable_head_rows': 2,
            'head_row_delta_l2': float(torch.linalg.vector_norm(rows.detach()-original)),
            'me_raser_reproduction': False}


@torch.no_grad()
def evaluate(model, tokenizer, token_ids, source, eval_texts, baseline_predictions=None):
    model.eval()
    historical, final, wrong = [], [], 0
    prompts = source['prompts']
    for start in range(0, len(prompts), 16):
        x = tokenizer(prompts[start:start+16], return_tensors='pt', padding=True,
                      truncation=True, max_length=40, return_token_type_ids=False)
        h = model.model(**x, use_cache=False).last_hidden_state
        old, last = detection_positions(x['attention_mask'])
        wrong += int((old != last).sum())
        rows = torch.arange(len(last))
        for index, target in ((old, historical), (last, final)):
            chosen = h[rows, index]
            if isinstance(model.lm_head, nn.Linear):
                bias = model.lm_head.bias
                pair = F.linear(chosen, model.lm_head.weight[token_ids],
                                bias[token_ids] if bias is not None else None)
            else:
                pair = model.lm_head(chosen)[:, token_ids]
            target.extend((pair[:, 1]-pair[:, 0]).float().tolist())
    def decode(values, only_heldout=False):
        arr = np.asarray(values).reshape(256, 3)
        bits = (arr[:, 1:].mean(1) >= 0).astype(np.int8) if only_heldout else aggregate(values, 3)
        scores = source['tardos'].scores(source['words'], bits)
        return {'matched_per_recipient': (np.asarray(source['words']) == bits).sum(1).tolist(),
                'scores': scores.tolist(), 'threshold': source['tardos'].threshold,
                'hits': [source['ids'][i] for i in source['tardos'].decode(source['words'], bits)]}
    total_loss = 0.; tokens = 0; correct = 0; predictions = []
    for i, text in enumerate(eval_texts):
        x = tokenizer(text, return_tensors='pt', truncation=True, max_length=64,
                      return_token_type_ids=False)
        out = model(**x, use_cache=False).logits[:, :-1]
        labels = x['input_ids'][:, 1:]
        mask = x['attention_mask'][:, 1:].bool()
        total_loss += float(F.cross_entropy(out[mask], labels[mask], reduction='sum'))
        pred = out.argmax(-1)[mask].cpu()
        predictions.append(pred)
        tokens += int(mask.sum())
        if baseline_predictions is not None:
            correct += int((pred == baseline_predictions[i]).sum())
    return {'historical': decode(historical), 'actual_last': decode(final),
            'actual_last_two_unexposed_prefixes': decode(final, True),
            'historical_not_last_count': wrong,
            'quality': {'perplexity': math.exp(total_loss/tokens), 'tokens': tokens,
                        'windows': len(eval_texts),
                        'agreement_vs_original': correct/tokens if baseline_predictions is not None else 1.0}}, predictions


def run_one(args):
    torch.set_num_threads(args.threads)
    source = verify_source(args.run_dir.resolve(), args.recipient)
    windows, corpus_hash = corpus_windows(args.corpus)
    if corpus_hash != source['source_metrics'].get('text_corpus_sha256'):
        raise ValueError('corpus mismatch')
    out = args.output.resolve() / f'U{args.recipient}'
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = {'status': 'running', 'recipient': args.recipient,
              'source_zip_sha256': source['payload_sha256'], 'corpus_sha256': corpus_hash,
              'trust': 'same-run public keys; internal consistency only, not independent identity authentication',
              'eval_windows': list(range(384, 400)), 'train_windows': list(range(448, 480)),
              'same_text_corpus': True, 'probe_reuse': 'all 768 probes were used when embedding',
              'baseline_source': 'SmolLM2 Q4_1 dequantized to fp32 before fingerprint embedding',
              'seed': 20261111, 'branches': {}, 'errors': {}}
    dump(out/'metrics.json', result)
    with tempfile.TemporaryDirectory(prefix='source-', dir=out) as temp:
        model_dir = _extract_model_archive(source['payload'], temp)
        baseline_predictions = None
        scenarios = ['baseline', 'row_int8_backbone', 'row_int8_with_head',
                     'prune10', 'prune30', 'structured_mlp10',
                     'full32', 'lora_r4_32', 'lora_r8_32',
                     'known_probe_erase', 'full32_then_row_sim', 'prune10_then_lora_r8']
        for scenario in scenarios:
            print(f'U{args.recipient} START {scenario}', flush=True)
            start = time.perf_counter()
            try:
                model, tokenizer, token_ids = load_local(model_dir, args.cache, args.threads)
                details = {}
                if scenario.startswith('row_int8_'):
                    details = pack_row_int8(model, scenario.endswith('with_head'), out/f'{scenario}.pt')
                elif scenario == 'prune10': details = prune_magnitude(model, .10)
                elif scenario == 'prune30': details = prune_magnitude(model, .30)
                elif scenario == 'structured_mlp10': details = prune_structured_mlp(model, .10)
                elif scenario == 'known_probe_erase':
                    details = erase_known_probes(model, tokenizer, source['prompts'],
                                                 source['words'][args.recipient-1], token_ids)
                elif scenario in ('full32', 'full32_then_row_sim'):
                    details['finetune'] = train(model, tokenizer, windows[448:480], 'full',
                        steps=32, lr=2e-5, max_length=64, seed=20261111, rank=8, alpha=16)
                    if scenario == 'full32_then_row_sim': details['quantization'] = apply_row_simulation(model)
                elif scenario in ('lora_r4_32', 'lora_r8_32', 'prune10_then_lora_r8'):
                    if scenario == 'prune10_then_lora_r8': details['pruning'] = prune_magnitude(model, .10)
                    rank = 4 if scenario == 'lora_r4_32' else 8
                    details['finetune'] = train(model, tokenizer, windows[448:480], 'lora',
                        steps=32, lr=2e-4, max_length=64, seed=20261111, rank=rank, alpha=2*rank)
                metrics, predictions = evaluate(model, tokenizer, token_ids, source, windows[384:400], baseline_predictions)
                if scenario == 'baseline': baseline_predictions = predictions
                if scenario != 'baseline':
                    base = result['branches']['baseline']['quality']['perplexity']
                    metrics['quality']['ppl_relative_pct'] = 100*(metrics['quality']['perplexity']/base-1)
                    metrics['quality']['demo_utility_pass'] = (metrics['quality']['ppl_relative_pct'] <= 10 and
                                                               metrics['quality']['agreement_vs_original'] >= .9)
                metrics['details'] = details
                metrics['elapsed_seconds'] = time.perf_counter()-start
                result['branches'][scenario] = metrics
                dump(out/'metrics.json', result)
                print('RESULT',json.dumps({'user':args.recipient,'scenario':scenario,
                      'last':metrics['actual_last'],'quality':metrics['quality']},ensure_ascii=False),flush=True)
            except Exception as exc:
                result['errors'][scenario] = f'{type(exc).__name__}: {exc}'
                dump(out/'metrics.json', result)
                print(f'ERROR U{args.recipient} {scenario}: {type(exc).__name__}: {exc}',flush=True)
                if scenario == 'baseline': raise
            finally:
                if 'model' in locals(): del model
                if 'tokenizer' in locals(): del tokenizer
                gc.collect()
    result['status'] = 'completed' if not result['errors'] else 'completed_with_errors'
    dump(out/'metrics.json', result)
    print(f'COMPLETE U{args.recipient} {result["status"]}',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipient', type=int, choices=(1,2,3), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, default=Path('work/multiprocess_real/run2'))
    p.add_argument('--corpus', type=Path, default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'))
    p.add_argument('--cache', type=Path, default=Path('.cache/huggingface'))
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    run_one(args)


if __name__ == '__main__':
    main()
