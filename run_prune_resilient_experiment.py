"""Research prototype: prune-aware embedding and a secret alternate readout.

It does not modify or re-sign issued copies. It saves protected experimental
head rows locally. The secret token-pair mapping and registered codebook must
not be exposed; with white-box access even these can still be removed.
"""
import argparse
import gc
import hashlib
import hmac
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
from safetensors.torch import save_file

from run_finetune_fingerprint_experiment import (_extract_model_archive, corpus_windows,
                                                  detection_positions, dump, load_local, verify_source)
from run_multiuser_robustness_suite import prune_magnitude, prune_structured_mlp
from tvmc.robustness import aggregate


def secret_pairs(key, vocab, excluded, bits=256, pair_count=16):
    """Select output-token pairs and per-bit pair slots from a private key."""
    choices = [i for i in range(vocab) if i not in excluded]
    ranked = sorted(choices, key=lambda i:hmac.new(key, b'token/'+str(i).encode(), hashlib.sha256).digest())
    tokens=ranked[:2*pair_count]
    slots=[int.from_bytes(hmac.new(key,b'slot/'+str(i).encode(),hashlib.sha256).digest()[:4], 'big') % pair_count
           for i in range(bits)]
    return tokens,slots


@torch.no_grad()
def prompt_features(model, tok, prompts):
    model.eval(); values=[]
    for start in range(0,len(prompts),16):
        x=tok(prompts[start:start+16],padding=True,truncation=True,max_length=40,
              return_tensors='pt',return_token_type_ids=False)
        h=model.model(**x,use_cache=False).last_hidden_state
        last=detection_positions(x['attention_mask'])[1]
        values.append(h[torch.arange(len(last)),last].detach().cpu())
    return torch.cat(values).reshape(256,3,-1)


@torch.no_grad()
def utility_features(model,tok,texts):
    pieces=[]
    for text in texts:
        x=tok(text,return_tensors='pt',truncation=True,max_length=64,return_token_type_ids=False)
        h=model.model(**x,use_cache=False).last_hidden_state[:,:-1]
        pieces.append(h.reshape(-1,h.shape[-1]).detach().cpu())
    return torch.cat(pieces)


def train_rows(original,features,utility,word,slots,steps=200,lr=.002):
    """Train only selected output-head rows against the supplied variants."""
    rows=nn.Parameter(original.detach().clone())
    opt=torch.optim.Adam([rows],lr=lr)
    y=torch.tensor(np.asarray(word)*2-1,dtype=torch.float32)
    slot=torch.tensor(slots,dtype=torch.long)
    # First two of each triple are exposed; third prefix is held out from this
    # adaptation, but all three were used by the prior watermark embedding.
    target=y[:,None].expand(-1,2).reshape(-1).repeat(len(features))
    pair=slot[:,None].expand(-1,2).reshape(-1).repeat(len(features))
    x=torch.cat([f[:,:2,:].reshape(-1,f.shape[-1]) for f in features])
    losses=[]
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        diff=rows[2*pair+1]-rows[2*pair]
        margin=(x*diff).sum(1)
        delta=rows-original
        loss=F.softplus(2.0-target*margin).mean()+.002*F.linear(utility,delta).square().mean()+.001*delta.square().mean()
        loss.backward();opt.step()
        if step in (0,steps-1):losses.append(float(loss.detach()))
    return rows.detach(),{'steps':steps,'learning_rate':lr,'training_variants':len(features),
                          'exposed_prefixes_per_bit':2,'final_prefix_held_out':True,
                          'heldout_prefix_was_in_original_embedding':True,'loss_first_last':losses,
                          'row_delta_l2':float(torch.linalg.vector_norm(rows-original))}


def decode(values,source,mode='all'):
    a=np.asarray(values,dtype=np.float64).reshape(256,3)
    observed=aggregate(values,3) if mode=='all' else (a[:,2]>=0).astype(np.int8)
    scores=source['tardos'].scores(source['words'],observed)
    return {'matches':(np.asarray(source['words'])==observed).sum(1).tolist(),
            'scores':scores.tolist(),'threshold':source['tardos'].threshold,
            'hits':[source['ids'][i] for i in source['tardos'].decode(source['words'],observed)]}


@torch.no_grad()
def get_trace(features,head,legacy_ids,alt_ids,slots,source):
    h=features.reshape(768,-1)
    primary=F.linear(h,head[legacy_ids])
    primary=(primary[:,1]-primary[:,0]).tolist()
    alt=F.linear(h,head[alt_ids]).reshape(256,3,len(alt_ids))
    slot=torch.tensor(slots)[:,None].expand(-1,3).reshape(256,3,1)
    negative=alt.gather(2,2*slot).squeeze(2)
    positive=alt.gather(2,2*slot+1).squeeze(2)
    values=(positive-negative).reshape(-1).tolist()
    return {'primary':decode(primary,source),'alternate':decode(values,source),
            'alternate_last_prefix':decode(values,source,'heldout'),
            'combined_hits':sorted(set(decode(primary,source)['hits']) | set(decode(values,source)['hits']))}


@torch.no_grad()
def quality(model,tok,texts,baseline_preds=None):
    model.eval(); loss=0.;count=0;equal=0;preds=[]
    for i,text in enumerate(texts):
        x=tok(text,return_tensors='pt',truncation=True,max_length=64,return_token_type_ids=False)
        z=model(**x,use_cache=False).logits[:,:-1]
        labels=x['input_ids'][:,1:];mask=x['attention_mask'][:,1:].bool()
        loss+=float(F.cross_entropy(z[mask],labels[mask],reduction='sum'))
        preds.append(z.argmax(-1)[mask].cpu());count+=int(mask.sum())
        if baseline_preds is not None:equal+=int((preds[-1]==baseline_preds[i]).sum())
    return {'perplexity':math.exp(loss/count),'tokens':count,
            'agreement_vs_unpruned_original':equal/count if baseline_preds is not None else 1.0},preds


def apply_prune(model,name):
    if name=='normal':return {'method':'none'}
    if name.startswith('mag'):
        return prune_magnitude(model,int(name[3:5])/100)
    if name.startswith('mlp'):
        return prune_structured_mlp(model,int(name[3:5])/100)
    raise ValueError(name)


def run(args):
    torch.set_num_threads(args.threads)
    key=args.key.read_bytes()
    if len(key)!=32:raise ValueError('expected 32-byte detector key')
    src=verify_source(args.run_dir.resolve(),args.recipient)
    windows,corpus_hash=corpus_windows(args.corpus)
    if corpus_hash!=src['source_metrics'].get('text_corpus_sha256'):raise ValueError('corpus mismatch')
    out=args.output.resolve()/f'U{args.recipient}'
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    result={'status':'running','recipient':args.recipient,'source_zip_sha256':src['payload_sha256'],
            'note':'experimental adaptation only; original signatures do not cover adjusted head rows',
            'trust':'same-run public keys, not independent authentication',
            'detection':'unpadded final token logits for secret selected pairs; requires requested token probabilities/logits',
            'security':'random local detector key; demonstration only, no collusion or adaptive attacker guarantee',
            'threshold':'same demo threshold reused; OR of two channels needs independent recalibration',
            'data':'same 768 original embedding probes; final prefix held out only from this adaptation',
            'train_variants':['normal','mag10','mag30'],
            'unseen_pruning_variants':['mag20','mag40','mlp02','mlp05'],
            'variants':{},'errors':{}}
    dump(out/'metrics.json',result)
    with tempfile.TemporaryDirectory(prefix='src-',dir=out) as tmp:
        base=_extract_model_archive(src['payload'],tmp)
        model,tok,legacy=load_local(base,args.cache,args.threads)
        if model.lm_head.bias is not None:raise ValueError('requires bias-free output head')
        alt,slots=secret_pairs(key,model.lm_head.weight.shape[0],set(tok.all_special_ids)|set(legacy))
        if len(set(alt))!=32:raise ValueError('duplicate alternate output rows')
        initial=model.lm_head.weight[alt].detach().clone()
        utility=utility_features(model,tok,windows[320:324])
        baseline_quality,baseline_preds=quality(model,tok,windows[384:400])
        feat={}
        for name in ('normal','mag10','mag30'):
            print('PRUNE FEATURES',args.recipient,name,flush=True)
            if name!='normal':
                del model;gc.collect();model,_,_=load_local(base,args.cache,args.threads)
                apply_prune(model,name)
            feat[name]=prompt_features(model,tok,src['prompts'])
        control,control_info=train_rows(initial,[feat['normal']],utility,src['words'][args.recipient-1],slots)
        robust,robust_info=train_rows(initial,[feat[x] for x in ('normal','mag10','mag30')],utility,
                                      src['words'][args.recipient-1],slots)
        result['training']={'normal_only':control_info,'prune_aware':robust_info,
                            'alternate_pairs':16,'alternate_head_rows':32,
                            'baseline_ppl':baseline_quality['perplexity']}
        rows_path=out/'experimental_head_rows.safetensors'
        save_file({'token_ids':torch.tensor(alt,dtype=torch.int64),'normal_only_rows':control,
                   'prune_aware_rows':robust},str(rows_path))
        rows_path.chmod(0o600)
        dump(out/'metrics.json',result)
        for name in ('normal','mag10','mag20','mag30','mag40','mlp02','mlp05'):
            print('PRUNE EVAL',args.recipient,name,flush=True);start=time.perf_counter()
            try:
                del model;gc.collect();model,_,_=load_local(base,args.cache,args.threads)
                info=apply_prune(model,name)
                h=feat.get(name)
                if h is None:h=prompt_features(model,tok,src['prompts'])
                original_quality,_=quality(model,tok,windows[384:400],baseline_preds)
                tests={}
                for label,rows in [('original',initial),('normal_only',control),('prune_aware',robust)]:
                    with torch.no_grad():model.lm_head.weight[alt]=rows
                    t=get_trace(h,model.lm_head.weight,legacy,alt,slots,src)
                    q,_=quality(model,tok,windows[384:400],baseline_preds)
                    q['relative_ppl_pct_vs_original_unpruned']=100*(q['perplexity']/baseline_quality['perplexity']-1)
                    q['relative_ppl_pct_vs_original_same_prune']=100*(q['perplexity']/original_quality['perplexity']-1)
                    tests[label]={'trace':t,'quality':q}
                result['variants'][name]={'pruning':info,'results':tests,'elapsed_seconds':time.perf_counter()-start}
                dump(out/'metrics.json',result)
                print('PRUNE RESULT',json.dumps({'user':args.recipient,'variant':name,
                    'primary':tests['original']['trace']['primary'],'control':tests['normal_only']['trace']['alternate'],
                    'robust':tests['prune_aware']['trace']['alternate'],
                    'ppl_original':original_quality['perplexity'],
                    'ppl_robust':tests['prune_aware']['quality']['perplexity']},ensure_ascii=False),flush=True)
            except Exception as e:
                result['errors'][name]=f'{type(e).__name__}: {e}';dump(out/'metrics.json',result)
                print('PRUNE ERROR',args.recipient,name,type(e).__name__,str(e),flush=True)
        # A knowledgeable attacker removes only the legacy 0/1 rows, then also
        # removes the secret alternate rows. These are targeted head attacks,
        # not evidence that arbitrary pruning can be resisted.
        del model;gc.collect();model,_,_=load_local(base,args.cache,args.threads)
        with torch.no_grad():model.lm_head.weight[alt]=robust
        for name,remove in [('target_legacy_rows',legacy),('target_both_channels',legacy+alt)]:
            with torch.no_grad():model.lm_head.weight[remove]=0
            t=get_trace(feat['normal'],model.lm_head.weight,legacy,alt,slots,src)
            q,_=quality(model,tok,windows[384:400],baseline_preds)
            q['relative_ppl_pct_vs_original_unpruned']=100*(q['perplexity']/baseline_quality['perplexity']-1)
            result['variants'][name]={'pruning':{'method':'targeted output rows set to zero','rows':len(remove)},
                                      'results':{'prune_aware':{'trace':t,'quality':q}}}
            dump(out/'metrics.json',result)
            print('PRUNE TARGETED',json.dumps({'user':args.recipient,'case':name,'trace':t,'quality':q},ensure_ascii=False),flush=True)
    result['status']='completed' if not result['errors'] else 'completed_with_errors';dump(out/'metrics.json',result)
    print('PRUNE COMPLETE',args.recipient,result['status'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipient',type=int,choices=(1,2,3),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--key',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,default=Path('work/multiprocess_real/run2'))
    p.add_argument('--corpus',type=Path,default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'))
    p.add_argument('--cache',type=Path,default=Path('.cache/huggingface'))
    p.add_argument('--threads',type=int,default=1)
    run(p.parse_args())

if __name__=='__main__':main()
