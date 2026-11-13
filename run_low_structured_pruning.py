"""Mild and single-layer structured pruning against the updated prototype."""
import argparse
import gc
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from run_finetune_fingerprint_experiment import _extract_model_archive,corpus_windows,dump,load_local,verify_source
from run_multiuser_robustness_suite import prune_structured_mlp
from run_prune_resilient_experiment import get_trace,prompt_features,quality,secret_pairs


def single_layer_mlp(model, index, fraction):
    """Zero coupled MLP neurons in one selected layer; do not resize tensors."""
    with torch.no_grad():
        mlp=model.model.layers[index].mlp
        n=mlp.gate_proj.weight.shape[0];k=max(1,int(n*fraction))
        importance=(mlp.gate_proj.weight.square().sum(1)+mlp.up_proj.weight.square().sum(1)+
                    mlp.down_proj.weight.square().sum(0))
        ids=torch.topk(importance,k,largest=False).indices
        mlp.gate_proj.weight[ids]=0;mlp.up_proj.weight[ids]=0;mlp.down_proj.weight[:,ids]=0
    return {'method':'one-layer structured MLP neuron zeroing', 'layer':index,'neurons_zeroed':k,
            'layer_neurons':n,'layer_fraction':k/n,'all_layer_fraction':k/(n*len(model.model.layers))}


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
            'note':'one model per user; same embedding probes, third prefix held out only from adaptation; output shape unchanged'}
    dump(out/'metrics.json',result)
    with tempfile.TemporaryDirectory(prefix='src-',dir=out) as tmp:
        base=_extract_model_archive(src['payload'],tmp)
        model,tok,legacy=load_local(base,a.cache,a.threads)
        orig,preds=quality(model,tok,windows[384:400]);key=a.key.read_bytes()
        alt,slots=secret_pairs(key,model.lm_head.weight.shape[0],set(tok.all_special_ids)|set(legacy))
        saved=load_file(str(a.structured/f'U{a.recipient}'/'structured_aware_head_rows.safetensors'))
        if alt!=saved['token_ids'].tolist():raise ValueError('key mismatch')
        for name,f in [('all_0.1pct',.001),('all_0.5pct',.005),('all_1pct',.01),('layer15_10pct',.1)]:
            print('MILD STRUCTURED START',a.recipient,name,flush=True)
            try:
                del model;gc.collect();model,_,_=load_local(base,a.cache,a.threads)
                info=single_layer_mlp(model,15,f) if name.startswith('layer') else prune_structured_mlp(model,f)
                h=prompt_features(model,tok,src['prompts'])
                before=get_trace(h,model.lm_head.weight,legacy,alt,slots,src)
                with torch.no_grad():model.lm_head.weight[alt]=saved['structured_aware_rows']
                after=get_trace(h,model.lm_head.weight,legacy,alt,slots,src)
                q,_=quality(model,tok,windows[384:400],preds)
                q['ppl_relative_pct']=100*(q['perplexity']/orig['perplexity']-1)
                result['cases'][name]={'pruning':info,'original_trace':before,'protected_trace':after,'quality':q}
                dump(out/'metrics.json',result)
                print('MILD STRUCTURED RESULT',json.dumps({'user':a.recipient,'name':name,'pruning':info,
                    'primary':before['primary'],'protected':after['alternate_last_prefix'],'quality':q},ensure_ascii=False),flush=True)
            except Exception as e:
                result['errors'][name]=f'{type(e).__name__}: {e}';dump(out/'metrics.json',result)
                print('MILD STRUCTURED ERROR',a.recipient,name,type(e).__name__,str(e),flush=True)
    result['status']='completed' if not result['errors'] else 'completed_with_errors';dump(out/'metrics.json',result)
    print('MILD STRUCTURED COMPLETE',a.recipient,result['status'],flush=True)

if __name__=='__main__':main()
