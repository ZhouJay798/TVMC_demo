"""Compare activation-aware MLP structured zeroing against weight-norm pruning.

Calibration uses ordinary corpus windows, no fingerprint probes or codeword.
The mask is computed once on the unpruned model, not iteratively recalibrated.
"""
import argparse
import gc
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from run_finetune_fingerprint_experiment import _extract_model_archive,corpus_windows,dump,load_local,verify_source
from run_prune_resilient_experiment import get_trace,prompt_features,quality,secret_pairs


@torch.no_grad()
def calibration_importance(model,tok,texts):
    """Mean squared neuron activation times squared down-projection norm."""
    layers=model.model.layers
    sums=[torch.zeros(l.mlp.down_proj.in_features,dtype=torch.float64) for l in layers]
    counts=[0 for _ in layers];handles=[]
    def hook(i):
        def collect(_module,inp):
            x=inp[0].reshape(-1,inp[0].shape[-1]).float()
            sums[i].add_(x.square().sum(0).double());counts[i]+=x.shape[0]
        return collect
    try:
        for i,layer in enumerate(layers):handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook(i)))
        model.eval()
        for text in texts:
            x=tok(text,return_tensors='pt',truncation=True,max_length=64,return_token_type_ids=False)
            model.model(**x,use_cache=False)
    finally:
        for h in handles:h.remove()
    if any(not c for c in counts):raise ValueError('empty calibration')
    return [(s/count)*layer.mlp.down_proj.weight.detach().float().square().sum(0).double()
            for s,count,layer in zip(sums,counts,layers)],counts


@torch.no_grad()
def prune_activation(model,importance,fraction):
    selected=total=0
    for layer,score in zip(model.model.layers,importance):
        mlp=layer.mlp;n=mlp.gate_proj.weight.shape[0];k=max(1,int(n*fraction))
        ids=torch.topk(score,k,largest=False).indices
        mlp.gate_proj.weight[ids]=0;mlp.up_proj.weight[ids]=0;mlp.down_proj.weight[:,ids]=0
        selected+=k;total+=n
    return {'method':'activation-aware structured MLP neuron zeroing, shapes unchanged',
            'requested_fraction':fraction,'neurons_zeroed':selected,'total_neurons':total,
            'actual_neuron_fraction':selected/total,'calibration_windows':list(range(288,304)),
            'output_head_pruned':False,'iterative_recalibration':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipient',type=int,choices=(1,2,3),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--key',type=Path,required=True)
    p.add_argument('--structured',type=Path,default=Path('work/prune_resilience/structured_run1'))
    p.add_argument('--run-dir',type=Path,default=Path('work/multiprocess_real/run2'))
    p.add_argument('--corpus',type=Path,default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'))
    p.add_argument('--cache',type=Path,default=Path('.cache/huggingface'))
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads)
    src=verify_source(a.run_dir.resolve(),a.recipient);windows,ch=corpus_windows(a.corpus)
    if ch!=src['source_metrics']['text_corpus_sha256']:raise ValueError('corpus mismatch')
    out=a.output.resolve()/f'U{a.recipient}';out.mkdir(mode=0o700,parents=True,exist_ok=False)
    result={'status':'running','recipient':a.recipient,'cases':{},'errors':{},
            'note':'calibration and eval use different windows of the same corpus; single model and seed; third prefix only held out from adaptation'}
    dump(out/'metrics.json',result)
    with tempfile.TemporaryDirectory(prefix='src-',dir=out) as tmp:
        base=_extract_model_archive(src['payload'],tmp)
        model,tok,legacy=load_local(base,a.cache,a.threads)
        orig,preds=quality(model,tok,windows[384:400]);key=a.key.read_bytes()
        alt,slots=secret_pairs(key,model.lm_head.weight.shape[0],set(tok.all_special_ids)|set(legacy))
        saved=load_file(str(a.structured/f'U{a.recipient}'/'structured_aware_head_rows.safetensors'))
        if alt!=saved['token_ids'].tolist():raise ValueError('key mismatch')
        print('ACTIVATION CALIBRATE',a.recipient,flush=True)
        importance,counts=calibration_importance(model,tok,windows[288:304]);result['calibration_tokens_per_layer']=counts
        for name,f in [('activation_0.5pct',.005),('activation_1pct',.01)]:
            print('ACTIVATION START',a.recipient,name,flush=True)
            try:
                del model;gc.collect();model,_,_=load_local(base,a.cache,a.threads)
                info=prune_activation(model,importance,f)
                h=prompt_features(model,tok,src['prompts'])
                before=get_trace(h,model.lm_head.weight,legacy,alt,slots,src)
                with torch.no_grad():model.lm_head.weight[alt]=saved['structured_aware_rows']
                after=get_trace(h,model.lm_head.weight,legacy,alt,slots,src)
                q,_=quality(model,tok,windows[384:400],preds)
                q['ppl_relative_pct']=100*(q['perplexity']/orig['perplexity']-1)
                result['cases'][name]={'pruning':info,'original_trace':before,'protected_trace':after,'quality':q}
                dump(out/'metrics.json',result)
                print('ACTIVATION RESULT',json.dumps({'user':a.recipient,'name':name,'pruning':info,
                   'primary':before['primary'],'heldout':after['alternate_last_prefix'],'quality':q},ensure_ascii=False),flush=True)
            except Exception as e:
                result['errors'][name]=f'{type(e).__name__}: {e}';dump(out/'metrics.json',result)
                print('ACTIVATION ERROR',a.recipient,name,type(e).__name__,str(e),flush=True)
    result['status']='completed' if not result['errors'] else 'completed_with_errors';dump(out/'metrics.json',result)
    print('ACTIVATION COMPLETE',a.recipient,result['status'],flush=True)

if __name__=='__main__':main()
