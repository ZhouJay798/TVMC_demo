"""Test actual multi-recipient model averages and public-base differencing.

Attribution outcomes are experimental, not proof of collusion resistance.
"""
import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from run_finetune_fingerprint_experiment import _extract_model_archive, corpus_windows, dump, load_local, verify_source
from run_multiuser_robustness_suite import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,default=Path('work/multiprocess_real/run2'))
    p.add_argument('--corpus',type=Path,default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'))
    p.add_argument('--clean-base',type=Path,default=Path('models/SmolLM2-135M-Instruct-Q4_1-converted'))
    p.add_argument('--cache',type=Path,default=Path('.cache/huggingface'))
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(args.threads)
    out=args.output.resolve();out.mkdir(mode=0o700,parents=True,exist_ok=False)
    sources=[verify_source(args.run_dir.resolve(),i) for i in (1,2,3)]
    windows,corpus_hash=corpus_windows(args.corpus)
    if any(corpus_hash != x['source_metrics'].get('text_corpus_sha256') for x in sources):
        raise ValueError('corpus mismatch')
    result={'status':'running','cases':{},'errors':{},'corpus_sha256':corpus_hash,
            'scope':'same registered 3-user codebook; 768 previously embedded probes; single seed and model',
            'trust':'same-run public keys, no independent trust root',
            'ppl_limit':'16 held-out windows from same corpus; not general task quality'}
    dump(out/'metrics.json',result)
    with tempfile.TemporaryDirectory(prefix='mix-',dir=out) as temporary:
        root=Path(temporary)
        paths=[]
        for i,s in enumerate(sources,1):
            target=root/f'U{i}'
            target.mkdir(mode=0o700)
            paths.append(_extract_model_archive(s['payload'],target))
        states=[load_file(str(x/'model.safetensors'),device='cpu') for x in paths]
        model,tok,tids=load_local(paths[0],args.cache,args.threads)
        clean,_,_=load_local(args.clean_base,args.cache,args.threads)
        clean_state={k:v.detach().clone() for k,v in clean.state_dict().items()}
        del clean;gc.collect()
        names=dict(model.named_parameters())
        for name in names:
            if any(name not in s for s in states) or name not in clean_state:
                raise ValueError(f'missing parameter {name}')
        scenarios=[('mean_U1_U2',(1,2),'mean'),('mean_U1_U3',(1,3),'mean'),
                   ('mean_U2_U3',(2,3),'mean'),('mean_U1_U2_U3',(1,2,3),'mean'),
                   ('diff_U1_minus_U2_plus_clean',(1,2),'diff'),
                   ('diff_U2_minus_U3_plus_clean',(2,3),'diff'),
                   ('diff_U3_minus_U1_plus_clean',(3,1),'diff')]
        for name,members,method in scenarios:
            print('COLLUSION START',name,flush=True);start=time.perf_counter()
            try:
                with torch.no_grad():
                    for key,param in names.items():
                        if method=='mean':
                            v=sum(states[i-1][key] for i in members)/len(members)
                        else:
                            a,b=members;v=states[a-1][key]-states[b-1][key]+clean_state[key]
                        param.copy_(v)
                metrics,_=evaluate(model,tok,tids,sources[0],windows[384:400])
                hits=metrics['actual_last']['hits']
                colluders={sources[0]['ids'][i-1] for i in members}
                metrics['contributors']=list(sorted(colluders))
                metrics['contributor_hit']=bool(colluders.intersection(hits))
                metrics['innocent_hits']=[x for x in hits if x not in colluders]
                metrics['method']=method
                metrics['elapsed_seconds']=time.perf_counter()-start
                result['cases'][name]=metrics
                dump(out/'metrics.json',result)
                print('COLLUSION RESULT',json.dumps({'case':name,'trace':metrics['actual_last'],
                      'ppl':metrics['quality']['perplexity'],'innocent_hits':metrics['innocent_hits']},ensure_ascii=False),flush=True)
            except Exception as e:
                result['errors'][name]=f'{type(e).__name__}: {e}'
                dump(out/'metrics.json',result)
                print('COLLUSION ERROR',name,type(e).__name__,str(e),flush=True)
    result['status']='completed' if not result['errors'] else 'completed_with_errors'
    dump(out/'metrics.json',result)
    print('COLLUSION COMPLETE',result['status'],flush=True)

if __name__=='__main__':main()
