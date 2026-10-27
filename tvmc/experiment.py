"""Measured experiments. Failures and unavailable model stages remain visible."""
import copy
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from .crypto import share_key, prove, recover, encrypt, decrypt, digest, verify_proof
from .tardos import Tardos, collude, wilson
from .bridge import Registry, Collector, OneShotBox, evidence_for, judge


def rejected(operation):
    try:
        operation()
    except Exception as exc:
        print(f'Expected rejection: {type(exc).__name__}: {exc}', flush=True)
        return True
    return False


def core_experiment(length=256, trials=2000):
    started = time.perf_counter()
    k, shares, public = share_key()
    context = {'sid': 'escrow-demo', 'epoch': 1, 'asset': 'SmolLM2-135M'}
    proofs = [prove(public, s, context) for s in shares]
    combinations = []
    for a, b in itertools.combinations(range(5), 2):
        ok = recover(public, [shares[a], shares[b]], [proofs[a], proofs[b]], context) == k
        combinations.append({'participants': [a+1, b+1], 'recovered': ok})
    ciphertext = encrypt(k, b'TVMC authenticated escrow test vector', context)
    tampered_proof = dict(proofs[0], zf=(proofs[0]['zf']+1))
    crypto = {
        'combinations': combinations,
        'all_10_recovered': all(x['recovered'] for x in combinations),
        'single_rejected': rejected(lambda: recover(public, shares[:1], proofs[:1], context)),
        'duplicate_rejected': rejected(lambda: recover(public, [shares[0]]*2, [proofs[0]]*2, context)),
        'tampered_share_rejected': rejected(lambda: recover(public,
            [replace(shares[0], value=(shares[0].value+1)), shares[1]], proofs[:2], context)),
        'tampered_ciphertext_rejected': rejected(lambda: decrypt(k, ciphertext[:-1]+bytes([ciphertext[-1]^1]), context)),
        'tampered_nizk_rejected': not verify_proof(public, tampered_proof, context),
        'cross_session_nizk_rejected': not verify_proof(public, proofs[0], dict(context, sid='wrong')),
        'all_5_nizk_verified': all(verify_proof(public, p, context) for p in proofs),
        'aes_roundtrip': decrypt(k, ciphertext, context) == b'TVMC authenticated escrow test vector',
    }
    tardos = Tardos(length=length)
    registry = Registry()
    records, words, openings = [], [], []
    # Consumer labels 1..5 identify released sessions, not convicted custodians.
    # Every session has an explicitly registered authorized pair.
    for i in range(5):
        rec, word, opening = registry.register(f'session-{i+1}', 1, [i+1, (i+1)%5+1],
                    public, tardos, expires=2_000_000_000, consumer=i+1)
        records.append(rec)
        words.append(word)
        openings.append(opening)
    words = np.array(words)
    singles = [{'user': i+1, 'decoded': [j+1 for j in tardos.decode(words, words[i])],
                'hit': i in tardos.decode(words, words[i])} for i in range(5)]
    rng = np.random.default_rng(314159)
    pairs = []
    for a, b in itertools.combinations(range(5), 2):
        for strategy in ('interleave', 'and', 'or'):
            y = collude(words[a], words[b], strategy, rng)
            decoded = tardos.decode(words, y)
            pairs.append({'coalition': [a+1, b+1], 'strategy': strategy,
                          'decoded': [i+1 for i in decoded], 'hit': bool({a,b}&set(decoded)),
                          'innocent_accused': len(set(decoded)-{a,b})})
    # Independent codebook per trial; thresholds are never tuned to results.
    false_count = innocent_count = scenario_false = hit_count = 0
    for run in range(trials):
        t = Tardos(length=length, seed=100_000+run)
        ws = np.array([t.encode(f'monte-carlo-user-{i}') for i in range(5)])
        a,b = rng.choice(5, 2, replace=False)
        strategy = ('interleave', 'and', 'or')[run % 3]
        decoded = set(t.decode(ws, collude(ws[a], ws[b], strategy, rng)))
        innocent = len(decoded-{int(a),int(b)})
        false_count += innocent
        innocent_count += 3
        scenario_false += innocent > 0
        hit_count += bool(decoded & {int(a),int(b)})
    code_metrics = {
        'n': 5, 'c': 2, 'length': length, 'threshold': tardos.threshold,
        'epsilon_target': .01, 'single': singles, 'single_hit_rate': sum(x['hit'] for x in singles)/5,
        'pair_cases': pairs, 'pair_hit_rate': sum(x['hit'] for x in pairs)/len(pairs),
        'all_pair_strategies_hit': all(x['hit'] for x in pairs),
        'monte_carlo': {'trials': trials, 'hits': hit_count, 'hit_rate': hit_count/trials,
                       'innocent_accusations': false_count, 'innocent_opportunities': innocent_count,
                       'false_accusation_rate': false_count/innocent_count,
                       'scenarios_with_false_accusation': scenario_false,
                       'scenario_false_rate_ci95': wilson(scenario_false, trials),
                       'empirical_fpr_le_1pct': false_count/innocent_count <= .01,
                       'scope': 'independent codebooks; marking assumption; 3 strategies, not all attacks'},
    }
    now = 1_800_000_000  # explicit logical verification time, no wall-clock expiration race
    collector = Collector()
    receipt, signature = collector.collect(OneShotBox(lambda _: words[0]), list(range(length)),
                                          'batch', 'core-fixture', 'challenge-1')
    evidence = evidence_for(records[0], words[0], openings[0], receipt, signature)
    def verify(e, reg=registry, at=now):
        return judge(reg, public, tardos, e, collector.public_key, 'core-fixture', 'challenge-1', at)
    wrong_seed = dict(evidence, seed='wrong')
    wrong_cm = dict(evidence, cm_w='wrong')
    fragment = copy.deepcopy(evidence)
    fragment['receipt']['observed'] = fragment['receipt']['observed'][:length//4]
    forged = copy.deepcopy(evidence)
    forged['receipt']['observed'][0] ^= 1
    # Real, signed fragment also fails coverage (not only the receipt signature).
    short_receipt, short_sig = collector.collect(OneShotBox(lambda _: words[0][:length//4]),
             list(range(length//4)), 'batch', 'core-fixture', 'challenge-1')
    signed_fragment = evidence_for(records[0], words[0], openings[0], short_receipt, short_sig)
    bridge = {'valid_accepted': verify(evidence), 'seed_mismatch_rejected': not verify(wrong_seed),
              'cm_mismatch_rejected': not verify(wrong_cm),
              'missing_reg_rejected': not verify(evidence, Registry()),
              'expired_rejected': not verify(evidence, at=2_000_000_001),
              'unsigned_fragment_frame_rejected': not verify(fragment),
              'signed_fragment_frame_rejected': not verify(signed_fragment),
              'edited_receipt_rejected': not verify(forged),
              'replay_nonce_rejected': not judge(registry, public, tardos, evidence,
                 collector.public_key, 'core-fixture', 'another-challenge', now),
              'scope': 'trusted registry and collector; does not prove endpoint provenance from bits alone'}
    metrics = {'crypto': crypto, 'tardos': code_metrics, 'bridge': bridge,
               'seconds': time.perf_counter()-started}
    state = {'key': k, 'shares': shares, 'public': public, 'tardos': tardos, 'registry': registry,
             'words': words, 'records': records, 'openings': openings, 'now': now}
    print('Core results:', json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics, state


def model_experiment(args, state, output):
    import torch
    from huggingface_hub import snapshot_download
    from .model import ModelLab, MODEL_ID, filtered_predict
    source = Path(args.model_path) if args.model_path else Path(snapshot_download(
        MODEL_ID, revision=args.revision, cache_dir=str(args.cache), local_files_only=args.offline,
        allow_patterns=['*.json', 'model.safetensors', 'merges.txt', 'vocab.json'], max_workers=1))
    weights = source / 'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(f'{weights}: download a complete SmolLM2-135M safetensors checkpoint')
    # Bind session proofs and authenticated ciphertext to the actual full model asset.
    import hashlib
    import shutil
    raw = weights.read_bytes()
    asset_hash = hashlib.sha256(raw).hexdigest()
    aad = {'sid': 'session-1', 'epoch': 1, 'sha256': asset_hash}
    encrypted = encrypt(state['key'], raw, aad)
    (output / 'model.aesgcm').write_bytes(encrypted)
    proofs = [prove(state['public'], s, aad) for s in state['shares'][:2]]
    recovered = recover(state['public'], state['shares'][:2], proofs, aad)
    decrypted = decrypt(recovered, encrypted, aad)
    if hashlib.sha256(decrypted).hexdigest() != asset_hash:
        raise ValueError('decrypted model checksum mismatch')
    local = output / 'decrypted_model'
    local.mkdir()
    for p in source.iterdir():
        if p.suffix in ('.json', '.txt'):
            shutil.copyfile(p, local / p.name)
    (local / 'model.safetensors').write_bytes(decrypted)
    del raw, encrypted, decrypted
    lab = ModelLab(str(local), args.cache, args.device, args.threads, local_only=True)
    words, tardos = state['words'], state['tardos']
    prompts = lab.triggers(tardos.master, tardos.length)
    features = lab.features(prompts)
    embeddings, adapters, observed_words = [], [], []
    progress = {'status': 'running', 'embeddings': embeddings, 'attacks': {}}
    def save_progress():
        (output/'model_progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
    base_utility = lab.clean_utility
    def measured(observed, utility, user=0):
        decoded = tardos.decode(words, observed)
        return {'bit_accuracy': float(np.mean(observed == words[user])),
                'fully_readable': bool(np.array_equal(observed, words[user])),
                'decoded_users': [i+1 for i in decoded], 'fingerprint_survives': user in decoded,
                'innocent_accused': len(set(decoded)-{user}), 'utility': utility,
                'relative_perplexity_increase': utility['perplexity']/base_utility['perplexity']-1,
                'utility_pass': utility['perplexity']/base_utility['perplexity'] <= 1.10
                                and utility['next_token_agreement'] >= .90}
    for i, word in enumerate(words):
        print(f'Embedding real SmolLM2 LoRA user={i+1}', flush=True)
        training = lab.embed(features, word, steps=args.steps)
        adapter = lab.snapshot()
        adapters.append(adapter)
        torch.save(adapter, output / f'user_{i+1}_lora.pt')
        observed = lab.predict(prompts)  # real end-to-end model forward, no cached decoding
        observed_words.append(observed)
        embeddings.append(dict(user=i+1, training=training, **measured(observed, lab.utility(), i)))
        save_progress()
        print('Embedding result:', json.dumps(embeddings[-1]), flush=True)
    # End-to-end evidence uses the model's actual predictions, including errors.
    lab.restore(adapters[0])
    collector = Collector()
    protocols, e2e = {}, None
    for protocol in ('batch', 'stream', 'single'):
        box = OneShotBox(lab.predict)
        receipt, signature = collector.collect(box, prompts, protocol, 'local-smollm2-user1', protocol)
        ev = evidence_for(state['records'][0], words[0], state['openings'][0], receipt, signature)
        accepted = judge(state['registry'], state['public'], tardos, ev, collector.public_key,
                         'local-smollm2-user1', protocol, state['now'])
        decoded = tardos.decode(words, receipt['observed']) if len(receipt['observed']) == tardos.length else []
        protocols[protocol] = {'requests': 1, 'returned_symbols': len(receipt['observed']),
                              'judge_accepted': accepted, 'decoded_users': [i+1 for i in decoded],
                              'second_request_rejected': rejected(lambda: box.query(prompts, protocol))}
        if protocol == 'batch':
            from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
            public_context = {
                'public': asdict(state['public']),
                'records': [asdict(r) for r in state['records']],
                'tardos_p': tardos.p.tolist(), 'threshold': tardos.threshold,
                'collector_key': collector.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
                'endpoint': 'local-smollm2-user1', 'nonce': protocol, 'now': state['now'],
                'key_proofs': proofs, 'key_context': aad,
            }
            (output/'public_context.json').write_text(json.dumps(public_context, indent=2), encoding='utf-8')
            e2e = {'key_layer': {'verified_participants': [1, 2], 'asset_decrypted': True,
                                  'sha256': asset_hash},
                   'deployment_layer': {'sessions': [state['records'][i].sid for i in decoded],
                                        'consumers': [i+1 for i in decoded], 'judge_accepted': accepted},
                   'bridge_layer': {'binding_accepted': accepted,
                                     'attribution_scope': 'session/consumer; participating custodians are not proven leakers'},
                   'status': 'passed' if accepted and 0 in decoded else 'failed',
                   'failure_layer': None if accepted and 0 in decoded else 'deployment_fingerprint'}
            (output/'public_evidence.json').write_text(json.dumps(ev, indent=2), encoding='utf-8')
    from .public_audit import audit_releases
    public_audit = audit_releases(lab, adapters, prompts, state, asset_hash, output)
    progress['public_audit'] = public_audit
    save_progress()
    # Actual coalition outputs from the five independently trained model adapters.
    model_pairs, rng = [], np.random.default_rng(2718)
    for a,b in itertools.combinations(range(5), 2):
        for strategy in ('interleave', 'and', 'or'):
            decoded = tardos.decode(words, collude(observed_words[a], observed_words[b], strategy, rng))
            model_pairs.append({'coalition':[a+1,b+1], 'strategy':strategy,
                                'hit':bool(set(decoded)&{a,b}), 'decoded_users':[i+1 for i in decoded],
                                'innocent_accused':len(set(decoded)-{a,b})})
    attacks = {}
    progress.update(protocols=protocols, end_to_end=e2e, model_coalitions=model_pairs, attacks=attacks)
    save_progress()
    for attack in ('finetuning', 'quantization', 'pruning', 'mixing', 'filtering', 'distillation'):
        print(f'Running actual attack: {attack}', flush=True)
        lab.restore(adapters[0])
        started = time.perf_counter()
        changed = []
        details = {}
        try:
            if attack == 'finetuning':
                details = {'steps': args.attack_steps, 'loss': lab.finetune(args.attack_steps),
                           'scope': 'clean-corpus output-head LoRA finetuning'}
            elif attack in ('quantization', 'pruning'):
                changed = lab.edit_transformer(attack, output/'weight_backup')
                details = {'scope': 'all transformer matrix weights except token embeddings',
                           'transform': 'symmetric per-row INT8 roundtrip (float inference)' if attack == 'quantization'
                                        else '30 percent magnitude pruning per matrix'}
            elif attack == 'mixing':
                # Mix the actual effective LoRA deltas, not A/B factors (which is incorrect).
                with torch.no_grad():
                    a1,b1 = adapters[0]['a'].to(lab.device),adapters[0]['b'].to(lab.device)
                    a2,b2 = adapters[1]['a'].to(lab.device),adapters[1]['b'].to(lab.device)
                    delta = .5*(b1@a1 + b2@a2)
                    u,s,vh = torch.linalg.svd(delta, full_matrices=False)
                    lab.head.a.zero_(); lab.head.b.zero_()
                    lab.head.a[:2].copy_(vh)
                    lab.head.b[:,:2].copy_(u*s)
                details = {'scope': '50/50 effective model-weight mixing, users 1 and 2'}
            elif attack == 'distillation':
                teacher = lab.logits(lab.train_inputs).cpu()
                lab.head.reset()  # clean base + independently initialized new adapter
                details = {'steps': args.attack_steps, 'kl_loss': lab.finetune(args.attack_steps, teacher),
                           'scope': 'clean pretrained SmolLM2 student, fresh output-head LoRA; KL distillation on clean inputs'}
            if attack == 'filtering':
                # Reject detected out-of-distribution triggers without consulting the target word.
                from .data import UTILITY
                observed = filtered_predict(lab.predict, prompts)
                clean_rejection = sum(x.startswith('TVMC probe') for x in UTILITY)/len(UTILITY)
                if clean_rejection:
                    raise ValueError('filter rejects utility corpus; unfiltered utility is invalid')
                utility = lab.utility()
                details = {'scope': 'input filter rejects prompts starting with TVMC probe',
                           'trigger_rejection_rate': float(np.mean(observed < 0)),
                           'clean_rejection_rate': clean_rejection}
            else:
                observed, utility = lab.predict(prompts), lab.utility()
            result = measured(observed, utility)
            if attack == 'mixing':
                ds = set(tardos.decode(words, observed))
                result['fingerprint_survives'] = bool(ds & {0,1})
                result['innocent_accused'] = len(ds-{0,1})
                result['coalition'] = [1,2]
            attacks[attack] = dict(status='completed', seconds=time.perf_counter()-started,
                                   parameters=details, **result)
            print('Attack result:', json.dumps(attacks[attack]), flush=True)
            save_progress()
        finally:
            lab.undo_transformer(changed)
            lab.restore(adapters[0])
    all_checks = (all(x['fully_readable'] and x['utility_pass'] for x in embeddings)
                  and all(x['hit'] and x['innocent_accused']==0 for x in model_pairs)
                  and e2e['status']=='passed' and all(r['passed'] for r in public_audit) and all(x['fingerprint_survives'] and x['utility_pass'] for x in attacks.values())
                  and all(x['judge_accepted'] for x in protocols.values()))
    model_fpr = {'innocent_accusations': sum(x['innocent_accused'] for x in model_pairs),
                 'innocent_opportunities': 3*len(model_pairs),
                 'scope': '30 correlated model-coalition cases; no population-level guarantee'}
    model_fpr['false_accusation_rate'] = model_fpr['innocent_accusations']/model_fpr['innocent_opportunities']
    model_fpr['empirical_fpr_le_1pct'] = model_fpr['false_accusation_rate'] <= .01
    return {'status':'completed', 'acceptance':'passed' if all_checks else 'failed',
            'model_id':MODEL_ID, 'requested_revision':args.revision,
            'resolved_revision': source.name if not args.model_path else 'local-path',
            'asset_sha256':asset_hash, 'device':lab.device, 'base_utility':base_utility,
            'embeddings':embeddings, 'protocols':protocols, 'model_coalitions':model_pairs,
            'attacks':attacks, 'public_audit':public_audit, 'end_to_end':e2e, 'model_false_accusation': model_fpr}
