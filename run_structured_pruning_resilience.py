"""Compare an additional structured-pruning-aware head with prior baselines.

This uses the same original embedding probes and evaluates a related third
prefix held out only from this new adaptation. It does not publish models.
"""
import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file,save_file

from run_finetune_fingerprint_experiment import _extract_model_archive,corpus_windows,dump,load_local,verify_source
from run_prune_resilient_experiment import (apply_prune,get_trace,prompt_features,quality,
                                            secret_pairs,train_rows,utility_features)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipient',type=int,choices=(1,2,3),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--key',type=Path,required=True)
    p.add_argument('--prior',type=Path,default=Path('work/prune_resilience/run1'))
    p.add_argument('--run-dir',type=Path,default=Path('work/multiprocess_real/run2'))
    p.add_argument('--corpus',type=Path,default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'))
    p.add_argument('--cache',type=Path,default=Path('.cache/huggingface'))
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads)
    source=verify_source(a.run_dir.resolve(),a.recipient)
    windows,corpus_hash=corpus_windows(a.corpus)
    if corpus_hash!=source['source_metrics']['text_corpus_sha256']:raise ValueError('corpus mismatch')
    out=a.output.resolve()/f'U{a.recipient}';out.mkdir(mode=0o700,parents=True,exist_ok=False)
    result={'status':'running','recipient':a.recipient,'cases':{},'errors':{},
       'train_variants':['normal','mag10','mag30','mlp02','mlp05'],
       'unseen_variants':['mag20','mag40','mlp03','mlp07'],
       'note':'Third prefix held out only from this adjustment; all prompts seen by the original fingerprint embedding. Reusing single-channel threshold is experimental.',
       'model':'signed source internally checked with same-run keys; updated head is unsigned research prototype'}
    dump(out/'metrics.json',result)
    with tempfile.TemporaryDirectory(prefix='src-',dir=out) as tmp:
        path=_extract_model_archive(source['payload'],tmp)
        model,tok,legacy=load_local(path,a.cache,a.threads)
        key=a.key.read_bytes()
        alt,slots=secret_pairs(key,model.lm_head.weight.shape[0],set(tok.all_special_ids)|set(legacy))
        prior=load_file(str(a.prior/f'U{a.recipient}'/'experimental_head_rows.safetensors'))
        if alt!=prior['token_ids'].tolist():raise ValueError('key mismatch')
        initial=model.lm_head.weight[alt].detach().clone()
        utility=utility_features(model,tok,windows[320:324]);baseline,preds=quality(model,tok,windows[384:400])
        features={}
        for name in result['train_variants']:
            print('STRUCTURED FEATURES',a.recipient,name,flush=True)
            if name!='normal':
                del model;gc.collect();model,_,_=load_local(path,a.cache,a.threads);apply_prune(model,name)
            features[name]=prompt_features(model,tok,source['prompts'])
        rows,info=train_rows(initial,[features[n] for n in result['train_variants']],utility,
                             source['words'][a.recipient-1],slots,steps=300,lr=.002)
        result['training']=info
        rowfile=out/'structured_aware_head_rows.safetensors'
        save_file({'token_ids':torch.tensor(alt,dtype=torch.int64),'structured_aware_rows':rows},str(rowfile))
        rowfile.chmod(0o600);dump(out/'metrics.json',result)
        for name in ('normal','mag20','mag40','mlp02','mlp03','mlp05','mlp07'):
            print('STRUCTURED EVAL',a.recipient,name,flush=True);started=time.perf_counter()
            try:
                del model;gc.collect();model,_,_=load_local(path,a.cache,a.threads)
                details=apply_prune(model,name)
                h=features.get(name)
                if h is None:h=prompt_features(model,tok,source['prompts'])
                comparison={}
                for label,head in [('ordinary',prior['normal_only_rows']),
                                   ('magnitude_aware',prior['prune_aware_rows']),
                                   ('structured_aware',rows)]:
                    with torch.no_grad():model.lm_head.weight[alt]=head
                    trace=get_trace(h,model.lm_head.weight,legacy,alt,slots,source)
                    q,_=quality(model,tok,windows[384:400],preds)
                    q['ppl_relative_pct']=100*(q['perplexity']/baseline['perplexity']-1)
                    comparison[label]={'trace':trace,'quality':q}
                result['cases'][name]={'pruning':details,'strategies':comparison,'elapsed_seconds':time.perf_counter()-started}
                dump(out/'metrics.json',result)
                print('STRUCTURED RESULT',json.dumps({'user':a.recipient,'variant':name,
                  'holdout_ordinary':comparison['ordinary']['trace']['alternate_last_prefix'],
                  'holdout_magnitude':comparison['magnitude_aware']['trace']['alternate_last_prefix'],
                  'holdout_structured':comparison['structured_aware']['trace']['alternate_last_prefix'],
                  'ppl_structured':comparison['structured_aware']['quality']['perplexity']},ensure_ascii=False),flush=True)
            except Exception as e:
                result['errors'][name]=f'{type(e).__name__}: {e}';dump(out/'metrics.json',result)
                print('STRUCTURED ERROR',a.recipient,name,type(e).__name__,str(e),flush=True)
    result['status']='completed' if not result['errors'] else 'completed_with_errors'
    dump(out/'metrics.json',result);print('STRUCTURED COMPLETE',a.recipient,result['status'],flush=True)


if __name__=='__main__':main()
