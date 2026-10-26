#!/usr/bin/env python
"""Paired research benchmark. Does not overwrite the original Demo/baseline.
Exit 0: measured improvement gates passed, NOT all attacks solved.
Exit 2: gates failed. Exit 3: execution error. Raw failures are always retained.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime,timezone
from uuid import uuid4
from run_demo import Tee


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--cache',type=Path,default=Path(__file__).resolve().parent/'.cache'/'huggingface')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--test-offset',type=int,default=128,
                        help='fixed final holdout: 128; 0 and 64 were used in development')
    parser.add_argument('--output',type=Path,default=Path('outputs')/'robustness')
    args=parser.parse_args()
    os.environ.setdefault('HF_HOME',str(args.cache.resolve()))
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:6]
    out=args.output/run_id; out.mkdir(parents=True)
    metrics={'run_id':run_id,'status':'running','parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             'gates_predeclared':{'ppl_max_relative_increase':.10,'agreement_min':.90,'fpr_max':.01,
              'compression':'group_int8 usable; sparsegpt30 lower PPL than magnitude30 at same sparsity',
              'filter':'natural+redundancy better hit than old prefix triggers under adaptive NLL filter',
              'limits':'passing improvements does not mean hard-label distillation or strict single-input tracing solved'},
             'embeddings':[],'compression':{},'filtering':{},'distillation':{}}
    metrics['runtime']={'python':sys.version,**{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','pyarrow')}}
    root=Path(__file__).resolve().parent
    metrics['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [root/'run_robustness.py',root/'tvmc'/'robustness.py',root/'tvmc'/'model.py']}
    def save():
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    with (out/'demo_output.log').open('w',encoding='utf-8',buffering=1) as log:
      with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        started=time.perf_counter()
        try:
            import numpy as np
            import torch
            from huggingface_hub import snapshot_download
            from tvmc.model import ModelLab,MODEL_ID
            from tvmc.tardos import Tardos
            from tvmc.robustness import (corpus,utility,utility_result,hessians,compressed,
                 natural_triggers,margins,aggregate,training_cache,embed_natural,
                 student_distillation,input_nll,filtered_observation,MODEL_REVISION,DATA_REVISION)
            print('Research run',run_id,flush=True)
            source=args.model_path or Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,
                cache_dir=str(args.cache),local_files_only=args.offline,
                allow_patterns=['*.json','model.safetensors','merges.txt','vocab.json'],max_workers=1))
            data=corpus(args.cache,args.offline,args.test_offset)
            metrics['dataset']={'repo':'Salesforce/wikitext','revision':DATA_REVISION,
              'documents':{k:len(v) for k,v in data.items()},
              'hashes':{k:[hashlib.sha256(x.encode()).hexdigest() for x in v] for k,v in data.items()},
              'test_offset':args.test_offset,'separation':'document-disjoint; article/topic independence not guaranteed'}
            metrics['model']={'id':MODEL_ID,'revision':MODEL_REVISION,
                 'sha256':hashlib.sha256((source/'model.safetensors').read_bytes()).hexdigest()}
            lab=ModelLab(str(source),args.cache,args.device,args.threads,True)
            clean=lab.snapshot()
            base,preds=utility(lab,data['test'])
            metrics['clean_test']=base
            rewritten_test=[' '.join(x.lower().split()) for x in data['test']]
            rewritten_base,rewritten_preds=utility(lab,rewritten_test)
            metrics['rewritten_clean_test']=rewritten_base
            metrics['legacy_clean']=lab.clean_utility
            print('Clean test',base,flush=True)
            t=Tardos(); words=np.array([t.encode(f'robust-user-{i+1}') for i in range(5)])
            old_prompts=lab.triggers(t.master,256)
            new_prompts=natural_triggers(lab,data['triggers'],3)
            (out/'benchmark_codebook.json').write_text(json.dumps({'words':words.tolist(),'seed':2026,'threshold':t.threshold}),encoding='utf-8')
            old_features=lab.features(old_prompts)
            new_features=lab.features(new_prompts)
            cal_cache=training_cache(lab,data['calibration'])
            distill_cache=training_cache(lab,data['distillation'])
            def observation(prompts,r): return aggregate(margins(lab,prompts),r)
            def assess(bits,user=0):
                decoded=t.decode(words,bits)
                return {'bit_accuracy':float(np.mean(bits==words[user])),
                    'observed_bits':np.asarray(bits).astype(int).tolist(),
                    'coverage':float(np.mean(bits>=0)), 'decoded_users':[x+1 for x in decoded],
                    'fingerprint_survives':user in decoded,'fully_readable':bool(np.array_equal(bits,words[user])),
                    'innocent_accused':len(set(decoded)-{user})}
            def measure(prompts,r,user=0):
                u,_=utility(lab,data['test'],preds)
                return dict(assess(observation(prompts,r),user),utility=utility_result(u,base))
            states={}
            for profile in ('original','natural_redundant'):
                states[profile]=[]
                for user in range(5):
                    print('Embedding',profile,'user',user+1,flush=True)
                    if profile=='original':
                        training=lab.embed(old_features,words[user],800)
                        prompts,r=old_prompts,1
                    else:
                        training=embed_natural(lab,new_features,np.repeat(words[user],3),cal_cache)
                        prompts,r=new_prompts,3
                    states[profile].append(lab.snapshot())
                    torch.save(lab.snapshot(),out/f'{profile}_user{user+1}.pt')
                    measured=measure(prompts,r,user)
                    # Report r=1 ablation from the same model; no extra hidden queries.
                    if profile=='natural_redundant':
                        single=aggregate(margins(lab,new_prompts[1::3]),1)
                        measured['single_trigger_ablation']=assess(single,user)
                    row=dict(profile=profile,user=user+1,training=training,**measured)
                    metrics['embeddings'].append(row); print(json.dumps(row),flush=True); save()
            lab.restore(clean)
            print('Collecting Hessians on clean non-trigger calibration corpus',flush=True)
            grams=hessians(lab,data['calibration'])
            for method in ('row_int8','group_int8','magnitude30','sparsegpt30','sparsegpt_mlp30'):
                print('Compression',method,flush=True)
                lab.restore(clean)
                with compressed(lab,method,grams) as info:
                    clean_u,_=utility(lab,data['test'],preds)
                    legacy_clean=lab.utility()
                    entry={'parameters':info,'clean_control':utility_result(clean_u,base),
                           'legacy_clean_control':utility_result(legacy_clean,lab.clean_utility)}
                    for profile,prompts,r in [('original',old_prompts,1),('natural_redundant',new_prompts,3)]:
                        lab.restore(states[profile][0])
                        entry[profile]=measure(prompts,r)
                        entry[profile]['legacy_utility']=utility_result(lab.utility(),lab.clean_utility)
                    metrics['compression'][method]=entry
                    print(json.dumps(entry),flush=True); save()
            del grams
            for profile,prompts,r in [('original',old_prompts,1),('natural_redundant',new_prompts,3)]:
                lab.restore(states[profile][0])
                # Filter knows no secret trigger list. Calibration threshold fixed at 95th percentile.
                cutoff=float(np.quantile(input_nll(lab,data['calibration']),.95))
                prompt_nll=input_nll(lab,prompts)
                clean_nll=input_nll(lab,data['test'])
                for filter_name in ('prefix','adaptive_nll','rewrite'):
                    if filter_name=='prefix':
                        keep=np.array([not x.startswith('TVMC probe') for x in prompts])
                        y=filtered_observation(lambda xs:margins(lab,xs),prompts,keep,r); clean_reject=0.
                    elif filter_name=='adaptive_nll':
                        keep=prompt_nll<=cutoff
                        y=filtered_observation(lambda xs:margins(lab,xs),prompts,keep,r)
                        clean_reject=float(np.mean(clean_nll>cutoff))
                    else:
                        # An attacker may normalize all prompts; no secret-code-aware rewrite.
                        rewritten=[' '.join(x.lower().split()) for x in prompts]
                        y=observation(rewritten,r); keep=np.ones(len(prompts),dtype=bool); clean_reject=0.
                    metrics['filtering'][f'{profile}/{filter_name}']=dict(assess(y),
                         threshold=cutoff if filter_name=='adaptive_nll' else None,trigger_rejection_rate=float(1-keep.mean()),
                         clean_rejection_rate=clean_reject,requests=len(prompts),
                         accepted_requests=int(keep.sum()),rejected_requests=int((~keep).sum()),
                         observed_code_bits=int((y>=0).sum()),
                         correct_code_bits=int((y==words[0]).sum()),
                         normal_requests=len(data['test']),
                         limitation='normal rejection measures availability; rewrite utility compares normalized text, not preserved semantics')
                    if filter_name=='rewrite':
                        rewrite_u,_=utility(lab,rewritten_test,rewritten_preds)
                        metrics['filtering'][f'{profile}/{filter_name}']['rewrite_utility']=utility_result(rewrite_u,rewritten_base)
                    print('Filtering',profile,filter_name,json.dumps(metrics['filtering'][f'{profile}/{filter_name}']),flush=True); save()
                for mode in ('soft','hard'):
                    lab.restore(states[profile][0])
                    training=student_distillation(lab,distill_cache,mode)
                    row=dict(training=training,**measure(prompts,r))
                    metrics['distillation'][f'{profile}/{mode}']=row
                    print('Distillation',profile,mode,json.dumps(row),flush=True); save()
            comp=metrics['compression']; filt=metrics['filtering']
            metrics['improvement_gates']={
                'group_int8_utility':comp['group_int8']['natural_redundant']['utility']['utility_pass'],
                'same_rate_pruning_better':comp['sparsegpt30']['natural_redundant']['utility']['perplexity']<comp['magnitude30']['natural_redundant']['utility']['perplexity'],
                'adaptive_filter_better':filt['natural_redundant/adaptive_nll']['fingerprint_survives'] and not filt['original/adaptive_nll']['fingerprint_survives'] and filt['natural_redundant/adaptive_nll']['clean_rejection_rate']<=.10,
                'natural_embedding_utility':all(x['utility']['utility_pass'] for x in metrics['embeddings'] if x['profile']=='natural_redundant'),
                'no_innocent_accusations':all(x['innocent_accused']==0 for x in list(filt.values())+list(metrics['distillation'].values()))}
            metrics['all_attacks_solved']=all(x['fingerprint_survives'] and x['utility']['utility_pass'] for k,x in metrics['distillation'].items() if k.startswith('natural_redundant/')) and all(x['fingerprint_survives'] for k,x in filt.items() if k.startswith('natural_redundant/')) and all(comp[m]['natural_redundant']['fingerprint_survives'] and comp[m]['natural_redundant']['utility']['utility_pass'] for m in ('group_int8','sparsegpt30'))
            exit_code=0 if all(metrics['improvement_gates'].values()) else 2
            metrics['status']='completed_improvements_verified' if exit_code==0 else 'completed_with_failed_gates'
        except Exception as exc:
            traceback.print_exc(); metrics.update(status='execution_error',error=f'{type(exc).__name__}: {exc}'); exit_code=3
        finally:
            metrics['seconds']=time.perf_counter()-started; metrics['exit_code']=exit_code
            save()
            from tvmc.research_report import write_report
            write_report(metrics,out)
            print('FINAL',metrics['status'],'exit',exit_code,'artifacts',out,flush=True)
    return exit_code


if __name__=='__main__': raise SystemExit(main())
