"""Paired full-parameter versus q/v LoRA fine-tuning of one delivered model.

Uses a locally saved recipient model, the registered codebook, and the *same*
768 embedding probes. This is a small, closed-box experiment, not a robustness
or attribution guarantee. U1's merged model is reloaded independently for each
branch; no modified weights are persisted. All output is sensitive local data.
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from tvmc.crypto import canonical
from tvmc.identity_gate import IdentityGate
from tvmc.personal_delivery import replica_seed
from tvmc.process_payloads import _extract_model_archive, file_sha256
from tvmc.robustness import aggregate
from tvmc.tardos import Tardos
from tvmc.trusted_registry import code_commitment, verify_issuance, verify_recipient_binding, verify_snapshot


class QVLoRA(nn.Module):
    """A standard additive rank-r update to an attention linear projection."""
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or type(rank) is not int or rank < 1 or alpha <= 0:
            raise ValueError('linear base, positive rank and alpha required')
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs):
        return self.base(inputs) + self.lora_b(self.lora_a(inputs)) * self.scale


def configure_training(model, mode, rank=8, alpha=16):
    """Unfreeze actual model weights or inject LoRA into Llama q/v projections."""
    if mode not in ('full', 'lora'):
        raise ValueError('mode must be full or lora')
    for parameter in model.parameters():
        parameter.requires_grad_(mode == 'full')
    targets = []
    if mode == 'lora':
        if getattr(model.config, 'model_type', None) != 'llama' or getattr(model.config, 'pretraining_tp', 1) != 1:
            raise ValueError('q/v LoRA is implemented only for Llama with pretraining_tp=1')
        for index, layer in enumerate(model.model.layers):
            for name in ('q_proj', 'v_proj'):
                base = getattr(layer.self_attn, name)
                if not isinstance(base, nn.Linear):
                    raise ValueError(f'unsupported attention projection {index}.{name}')
                setattr(layer.self_attn, name, QVLoRA(base, rank, alpha))
                targets.append(f'model.layers.{index}.self_attn.{name}')
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named:
        raise ValueError('no trainable parameters')
    if mode == 'lora' and (any('.lora_a.' not in name and '.lora_b.' not in name for name, _ in named) or
                           len(named) != 2 * len(targets)):
        raise ValueError('unexpected trainable non-LoRA parameter')
    if mode == 'full' and len(named) != sum(1 for _ in model.named_parameters()):
        raise ValueError('not all real model parameters are trainable')
    return {'target_modules': targets, 'trainable_names': [name for name, _ in named],
            'trainable_parameters': sum(p.numel() for _, p in named),
            'total_parameters': sum(p.numel() for p in model.parameters())}


def load_local(model_dir, cache, threads):
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=str(cache), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, cache_dir=str(cache), local_files_only=True,
                                                 torch_dtype=torch.float32).cpu()
    if getattr(model.config, 'model_type', None) != 'llama' or not hasattr(model, 'lm_head'):
        raise ValueError('this small experiment currently supports Llama causal models only')
    tokens = [tokenizer.encode(value, add_special_tokens=False) for value in ('0', '1')]
    if any(len(value) != 1 for value in tokens) or tokens[0] == tokens[1]:
        raise ValueError('0 and 1 must be distinct single tokens')
    return model, tokenizer, [value[0] for value in tokens]


def verify_source(root, recipient):
    """Verify saved records with same-run public keys (not independent anchors)."""
    def read(name):
        return json.loads((root / name).read_text(encoding='utf-8'))
    reg, signed, pre, issued, code = (read(name) for name in
        ('public_registration.json', 'signed_manifest.json', 'registry_precommit.json',
         'registry_issued.json', 'audit_codebook.json'))
    issuer_hex, registrar_hex = reg['issuer_public'], reg['registrar_public']
    if signed['issuer_public'] != issuer_hex or signed['registrar_public'] != registrar_hex:
        raise ValueError('inconsistent same-run public keys')
    issuer = Ed25519PublicKey.from_public_bytes(bytes.fromhex(issuer_hex))
    registrar = Ed25519PublicKey.from_public_bytes(bytes.fromhex(registrar_hex))
    ad = reg['admission']; gate = IdentityGate(reg['enrollment'])
    admission = gate.register(ad['sid'], ad['epoch'], ad['participants'], ad['base_sha256'], ad['nonce'], ad['proofs'])
    if canonical(admission.__dict__) != canonical(ad):
        raise ValueError('saved admission mismatch')
    manifest = signed['manifest']; signature = bytes.fromhex(signed['signature'])
    body = verify_issuance(pre, issued, registrar, manifest, signature, issuer, gate, admission, reg['recipients'])
    pre_body = verify_snapshot(pre, registrar)
    ids = [f'{admission.sid}-U{i}' for i in admission.participants]
    if code['replica_ids'] != ids or reg['replica_ids'] != ids or recipient not in admission.participants:
        raise ValueError('unexpected recipient or registered replica IDs')
    tardos = Tardos(n=len(ids)); scheme = pre_body['fingerprint_scheme']
    if (scheme['p'] != tardos.p.tolist() or scheme['threshold'] != tardos.threshold or
        code['p'] != scheme['p'] or code['threshold'] != scheme['threshold'] or len(code['words']) != len(ids)):
        raise ValueError('saved codebook parameters differ from signed registration')
    for rid, word, record in zip(ids, code['words'], reg['recipients']):
        checked = verify_recipient_binding(body, rid, record['recipient'], bytes.fromhex(record['public_key_der']))
        if (tardos.encode(replica_seed(admission, record)).tolist() != word or
            code_commitment(rid, word, code['openings'][rid]) != checked['code_commitment']):
            raise ValueError('fingerprint does not match its registered commitment')
    payload = root / 'custodians' / f'U{recipient}' / 'received_model.zip'
    matches = [entry for entry in manifest['entries'] if entry['context']['recipient'] == recipient]
    actual_hash = file_sha256(payload)
    if len(matches) != 1 or matches[0]['context']['payload_sha256'] != actual_hash:
        raise ValueError('recipient ZIP does not match the signed manifest')
    prompts = read('audit_prompts.json')
    if not isinstance(prompts, list) or len(prompts) != 768 or not all(isinstance(x, str) for x in prompts):
        raise ValueError('expected 768 saved training probes')
    return {'payload': payload, 'payload_sha256': actual_hash, 'ids': ids, 'words': code['words'],
            'prompts': prompts, 'tardos': tardos, 'source_metrics': read('metrics.json')}


def corpus_windows(path):
    raw = Path(path).read_bytes()
    text = raw.decode('utf-8')
    windows = [text[start:start + 1024].strip() for start in range(0, len(text) - 1024 + 1, 1024)]
    # The original splitter discards windows shorter than 240 after stripping;
    # reject rather than silently change the meaning of saved indices.
    if len(windows) < 449 or any(len(window) < 240 for window in windows):
        raise ValueError('corpus must provide at least 449 full nonempty 1024-character windows')
    return windows, hashlib.sha256(raw).hexdigest()


@torch.no_grad()
def detection_positions(mask):
    """Return the historical project index and the actual final non-pad index."""
    if (mask.ndim != 2 or not bool(((mask == 0) | (mask == 1)).all()) or
        bool((mask.sum(1) == 0).any())):
        raise ValueError('invalid attention mask')
    historical = mask.sum(1) - 1
    last = (torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) * mask).max(1).values
    return historical, last


@torch.no_grad()
def detect(model, tokenizer, token_ids, prompts, words, replica_ids, tardos):
    model.eval(); values=[]; last_values=[]; padded_count=wrong_count=on_padding=0
    for start in range(0, len(prompts), 16):
        inputs = tokenizer(prompts[start:start + 16], return_tensors='pt', padding=True,
                           truncation=True, max_length=40, return_token_type_ids=False)
        hidden = model.model(**inputs, use_cache=False).last_hidden_state
        mask = inputs['attention_mask']
        historical, last = detection_positions(mask)
        rows = torch.arange(len(last))
        padded_count += int((mask.sum(1) != mask.shape[1]).sum())
        wrong_count += int((historical != last).sum())
        on_padding += int((mask[rows, historical] == 0).sum())
        for index, output in ((historical, values), (last, last_values)):
            pair = model.lm_head(hidden[rows, index])[:, token_ids]
            output.extend((pair[:, 1] - pair[:, 0]).float().tolist())
    if len(values) != 768 or len(last_values) != 768 or not np.isfinite(values).all() or not np.isfinite(last_values).all():
        raise ValueError('incomplete or non-finite model responses')
    def decode(raw, description):
        observed = aggregate(raw, 3)
        scores = tardos.scores(words, observed)
        return {'score_differences': raw, 'observed_bits': observed.tolist(),
                'correct_bits_per_replica': [int((observed == w).sum()) for w in np.asarray(words)],
                'tardos_scores': scores.tolist(), 'threshold': tardos.threshold,
                'decoded_replicas': [replica_ids[i] for i in tardos.decode(words, observed)],
                'input_count': len(prompts), 'response': description, 'individual_leaker_proven': False}
    result = decode(values, 'historical attention_mask.sum-1 position: may not be final token with left padding')
    result['position_diagnostic'] = {'padding_side': tokenizer.padding_side, 'batch_size': 16,
        'padded_inputs': padded_count, 'historical_not_last_count': wrong_count,
        'historical_selected_padding_count': on_padding}
    result['correct_last_token'] = decode(last_values, 'actual final non-padding token: next-token 1 minus 0 logit')
    return result


@torch.no_grad()
def perplexity(model, tokenizer, texts, max_length):
    model.eval(); weighted_loss=0.; total_tokens=0
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length,
                           return_token_type_ids=False)
        count = int(inputs['attention_mask'][:, 1:].sum())
        if count < 1:
            raise ValueError('evaluation text too short')
        loss = model(**inputs, labels=inputs['input_ids'], use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError('non-finite evaluation loss')
        weighted_loss += float(loss) * count
        total_tokens += count
    return {'perplexity': math.exp(weighted_loss / total_tokens), 'tokens': total_tokens,
            'windows': len(texts), 'max_length': max_length}


def train(model, tokenizer, texts, mode, *, steps, lr, max_length, seed, rank, alpha):
    torch.manual_seed(seed)
    configuration = configure_training(model, mode, rank, alpha)
    q = model.model.layers[0].self_attn.q_proj
    first_q_before = (q.base.weight if mode == 'lora' else q.weight).detach().clone()
    head_before = model.lm_head.weight.detach()[:64].clone()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train(); started=time.perf_counter(); losses=[]; grad_names=set(); training_tokens=0
    for step in range(steps):
        inputs = tokenizer(texts[step % len(texts)], return_tensors='pt', truncation=True,
                           max_length=max_length, return_token_type_ids=False)
        training_tokens += int(inputs['attention_mask'][:, 1:].sum())
        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs, labels=inputs['input_ids'], use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError('non-finite training loss')
        loss.backward()
        if step == 0:
            grad_names = {name for name, param in model.named_parameters() if param.grad is not None}
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 4 == 0 or step + 1 == steps:
            print(f'{mode}: step {step + 1}/{steps}, loss={losses[-1]:.4f}', flush=True)
    q_after = model.model.layers[0].self_attn.q_proj
    q_base = q_after.base.weight if mode == 'lora' else q_after.weight
    q_diff = float(torch.linalg.vector_norm(q_base.detach() - first_q_before))
    head_diff = float(torch.linalg.vector_norm(model.lm_head.weight.detach()[:64] - head_before))
    lora_diff = (float(torch.linalg.vector_norm(q_after.lora_b.weight.detach() @ q_after.lora_a.weight.detach()) * q_after.scale)
                 if mode == 'lora' else None)
    if mode == 'full' and (q_diff <= 0 or head_diff <= 0):
        raise ValueError('full fine-tuning did not change both backbone q_proj and output head')
    if mode == 'lora' and (q_diff != 0 or head_diff != 0 or lora_diff <= 0):
        raise ValueError('LoRA base/head changed or LoRA update is zero')
    if mode == 'lora' and any('.lora_a.' not in name and '.lora_b.' not in name for name in grad_names):
        raise ValueError('unexpected frozen base gradients in LoRA mode')
    configuration.update(mode=mode, steps=steps, learning_rate=lr, weight_decay=0.0,
                         optimizer='AdamW', max_length=max_length, batch_size=1, seed=seed,
                         training_tokens=training_tokens,
                         losses=losses, first_loss=losses[0], last_loss=losses[-1],
                         elapsed_seconds=time.perf_counter()-started,
                         first_q_base_weight_delta_l2=q_diff, first_64_head_rows_delta_l2=head_diff,
                         first_q_effective_lora_delta_l2=lora_diff,
                         parameters_with_gradient_on_first_step=len(grad_names),
                         backbone_gradient_observed=any(name.startswith('model.layers.') for name in grad_names),
                         head_gradient_observed=any(name.startswith('lm_head.') for name in grad_names))
    return configuration


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    path.chmod(0o600)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=Path('work/multiprocess_real/run2'))
    p.add_argument('--corpus',type=Path,default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recipient',type=int,default=1,choices=(1,2,3))
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--full-lr',type=float,default=2e-5)
    p.add_argument('--lora-lr',type=float,default=2e-5)
    p.add_argument('--max-length',type=int,default=64)
    p.add_argument('--eval-windows',type=int,default=16)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--alpha',type=float,default=16.0)
    p.add_argument('--seed',type=int,default=20261111)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--cache',type=Path,default=Path('.cache/huggingface'))
    args=p.parse_args()
    if (not 1<=args.steps<=10000 or not 1<=args.eval_windows<=64 or args.max_length<2 or args.rank<1 or
        not math.isfinite(args.alpha) or args.alpha<=0 or args.threads<1 or
        not all(math.isfinite(lr) and lr>0 for lr in (args.full_lr,args.lora_lr))):
        p.error('invalid steps, learning rate, model/LoRA parameters or evaluation size')
    root=args.run_dir.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False,mode=0o700)
    print('VERIFYING signed source and registered codebook (same-run trust, not independent)',flush=True)
    source=verify_source(root,args.recipient)
    windows,corpus_sha=corpus_windows(args.corpus)
    expected_sha=source['source_metrics'].get('text_corpus_sha256')
    if expected_sha != corpus_sha:
        raise ValueError('the fine-tuning corpus must match the recorded original corpus SHA-256')
    training_indices=[448 + (i % (len(windows)-448)) for i in range(args.steps)]
    eval_indices=list(range(384,384+args.eval_windows))
    train_texts=[windows[i] for i in training_indices]
    eval_texts=[windows[i] for i in eval_indices]
    meta={'status':'running','scope':'one previously delivered model; trained and tested on the same 768 fingerprint probes',
          'model_scope':source['source_metrics']['model_scope'],'model_base_sha256':source['source_metrics']['base_sha256'],
          'model_source':source['source_metrics'].get('model_source'),
          'source_recipient':args.recipient,'source_replica_id':source['ids'][args.recipient-1],
          'source_zip_sha256':source['payload_sha256'],'source_run':str(root),'corpus_sha256':corpus_sha,
          'corpus_split':{'trigger_windows':'0..255','original_calibration_windows':'256..383',
                          'evaluation_window_indices':eval_indices,'fine_tune_window_indices':training_indices,
                          'window_characters':1024,'independent_documents':False},
          'configuration':{'steps':args.steps,'full_learning_rate':args.full_lr,'lora_learning_rate':args.lora_lr,'max_length':args.max_length,
                           'eval_windows':args.eval_windows,'seed':args.seed,'rank':args.rank,'alpha':args.alpha,
                           'threads':args.threads},
          'trust':'issuer/registrar signatures, bindings and source hash rechecked with public keys from the same run; not independent authentication',
          'branches':{}}
    dump(out/'metrics.json',meta)
    with tempfile.TemporaryDirectory(prefix='tvmc-finetune-',dir=out) as directory:
        model_dir=_extract_model_archive(source['payload'],directory)
        for mode in ('baseline','full','lora'):
            print('BRANCH',mode,flush=True)
            model,tokenizer,token_ids=load_local(model_dir,args.cache,args.threads)
            training=None
            if mode!='baseline':
                training=train(model,tokenizer,train_texts,mode,steps=args.steps,lr=args.full_lr if mode=='full' else args.lora_lr,
                               max_length=args.max_length,seed=args.seed,rank=args.rank,alpha=args.alpha)
            trace=detect(model,tokenizer,token_ids,source['prompts'],source['words'],source['ids'],source['tardos'])
            quality=perplexity(model,tokenizer,eval_texts,args.max_length)
            dump(out/f'{mode}_trace.json',trace)
            scores=trace['tardos_scores']; index=args.recipient-1
            summary={'source_bits_matched':trace['correct_bits_per_replica'][index],
                     'correct_bits_per_replica':trace['correct_bits_per_replica'],
                     'source_tardos_score':scores[index],'tardos_scores':scores,'threshold':trace['threshold'],
                     'decoded_replicas':trace['decoded_replicas'],'source_detected':source['ids'][index] in trace['decoded_replicas'],
                     'quality':quality,'training':training,
                     'position_diagnostic':trace['position_diagnostic'],
                     'correct_last_token':{
                         'source_bits_matched':trace['correct_last_token']['correct_bits_per_replica'][index],
                         'correct_bits_per_replica':trace['correct_last_token']['correct_bits_per_replica'],
                         'source_tardos_score':trace['correct_last_token']['tardos_scores'][index],
                         'tardos_scores':trace['correct_last_token']['tardos_scores'],
                         'threshold':trace['correct_last_token']['threshold'],
                         'decoded_replicas':trace['correct_last_token']['decoded_replicas'],
                         'source_detected':source['ids'][index] in trace['correct_last_token']['decoded_replicas']}}
            meta['branches'][mode]=summary
            if mode!='baseline':
                baseline=meta['branches']['baseline']
                summary['score_change_from_baseline']=summary['source_tardos_score']-baseline['source_tardos_score']
                summary['ppl_relative_change_pct']=100*(quality['perplexity']/baseline['quality']['perplexity']-1)
                base_trace=json.loads((out/'baseline_trace.json').read_text(encoding='utf-8'))
                summary['bits_changed_from_baseline']=int(np.count_nonzero(np.asarray(trace['observed_bits']) != np.asarray(base_trace['observed_bits'])))
            dump(out/'metrics.json',meta)
            print('RESULT',json.dumps({'mode':mode,'matched':summary['source_bits_matched'],
                  'decoded':summary['decoded_replicas'],'score':summary['source_tardos_score'],
                  'ppl':quality['perplexity']},ensure_ascii=False),flush=True)
            del model,tokenizer,trace
            gc.collect()
    meta['status']='completed'
    meta['limitations']=['One U1 model, one training run and a small number of steps per branch; no population-level success or robustness rate.',
      'Same 768 probes used to embed and detect; each three prefixes share a text window and are correlated.',
      'Historical project detection uses attention_mask.sum-1, which does not select the actual last token with left padding. Both historical and corrected positions are reported; fingerprints were embedded with the historical convention.',
      'Fine-tuning and evaluation use non-overlapping windows of the same Tiny Shakespeare file, not independent documents.',
      ('The baseline is a quantized-then-dequantized local model; not the original unquantized checkpoint.'
       if 'Q4_1' in str(meta.get('model_source')) else 'The exact model identity and provenance follow the original local run, not independent authentication.'),
      'Equal learning rate and steps do not equalize the strength of full and LoRA fine-tuning.',
      'LoRA adapters target attention q_proj/v_proj and remain active; adapters are not merged into saved model weights.',
      'No fine-tuned checkpoints are saved; reproduce them by rerunning the experiment.',
      'The registered codebook, probes and outputs are sensitive local research files; public keys from the same run are not independent trust anchors.',
      'The experiment provides copy-source evidence only and does not identify a real-world leaker.']
    dump(out/'metrics.json',meta)
    print('COMPLETE',json.dumps({'status':meta['status']},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
