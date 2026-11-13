"""Native llama.cpp GGUF quantization/CPU inference of delivered U1/U2/U3.

Uses the project's saved probe and codebook, so outputs are sensitive. The
actual GGUF model files are retained locally, not automatically published.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from llama_cpp import Llama
from transformers import AutoTokenizer

from run_finetune_fingerprint_experiment import corpus_windows, dump, verify_source, _extract_model_archive
from tvmc.robustness import aggregate


def command(args, logfile):
    with open(logfile, 'ab') as stream:
        subprocess.run([str(s) for s in args], check=True, stdout=stream, stderr=subprocess.STDOUT)


def inspect_gguf(path):
    r = GGUFReader(path)
    types = {}
    for t in r.tensors:
        key = str(t.tensor_type.name)
        types[key] = types.get(key, 0)+1
    special = {t.name: str(t.tensor_type.name) for t in r.tensors
               if t.name in ('output.weight','token_embd.weight','blk.0.attn_q.weight')}
    return {'tensor_type_counts': types, 'selected_tensor_types': special, 'file_bytes': path.stat().st_size}


def evaluate_gguf(path, tokenizer, source, eval_texts, threads):
    llm = Llama(model_path=str(path), n_ctx=64, n_batch=64, n_threads=threads,
                n_threads_batch=threads, n_gpu_layers=0, logits_all=True, verbose=False)
    ids0 = tokenizer.encode('0', add_special_tokens=False)
    ids1 = tokenizer.encode('1', add_special_tokens=False)
    if len(ids0) != 1 or len(ids1) != 1 or llm.n_vocab() != len(tokenizer):
        raise ValueError('tokenizer/vocabulary mismatch')
    i0,i1 = ids0[0], ids1[0]
    raw=[]
    for p in source['prompts']:
        tokens = tokenizer(p, truncation=True, max_length=40)['input_ids']
        llm.reset(); llm.eval(tokens)
        z = llm.scores[len(tokens)-1]
        raw.append(float(z[i1]-z[i0]))
    observed=aggregate(raw,3)
    tardos=source['tardos']
    scores=tardos.scores(source['words'],observed)
    trace={'matched_per_recipient':(np.asarray(source['words'])==observed).sum(1).tolist(),
           'scores':scores.tolist(),'threshold':tardos.threshold,
           'hits':[source['ids'][i] for i in tardos.decode(source['words'],observed)]}
    nll=0.; count=0; predictions=[]
    for text in eval_texts:
        tokens=tokenizer(text,truncation=True,max_length=64)['input_ids']
        llm.reset();llm.eval(tokens)
        z=np.asarray(llm.scores[:len(tokens)-1],dtype=np.float64)
        labels=np.asarray(tokens[1:]);mx=z.max(1)
        nll += float((mx+np.log(np.exp(z-mx[:,None]).sum(1))-z[np.arange(len(labels)),labels]).sum())
        count += len(labels)
        predictions.append(z.argmax(1).astype('int32').tolist())
    llm.close()
    return trace,{'perplexity':math.exp(nll/count),'tokens':count,'windows':len(eval_texts)},predictions


def run(args):
    out=args.output.resolve()/f'U{args.recipient}'
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    src=verify_source(args.run_dir.resolve(),args.recipient)
    windows,corpus_hash=corpus_windows(args.corpus)
    if corpus_hash != src['source_metrics'].get('text_corpus_sha256'):
        raise ValueError('corpus mismatch')
    extracted=out/'extracted'
    extracted.mkdir(mode=0o700)
    source_dir=_extract_model_archive(src['payload'],extracted)
    tok=AutoTokenizer.from_pretrained(source_dir,local_files_only=True)
    f32=out/'source-f32.gguf'
    command([sys.executable,args.llama_root/'convert_hf_to_gguf.py',source_dir,'--outtype','f32','--outfile',f32],out/'convert.log')
    result={'status':'running','recipient':args.recipient,'source_zip_sha256':src['payload_sha256'],
            'trust':'same-run anchors, not independent authentication',
            'method':'actual GGUF weights, native llama.cpp CPU inference; same 768 embedding probes; unpadded next-token logits',
            'corpus_sha256':corpus_hash,'eval_windows':list(range(384,400)),
            'model_source':'prior Q4_1 dequantized to fp32, then fingerprinted',
            'branches':{},'errors':{}}
    dump(out/'metrics.json',result)
    cases=[('baseline_f32',None,None),('group_q8_head_f32','Q8_0','keep'),
           ('group_q8_head_q8','Q8_0','q8_0'),('group_q4_0','Q4_0','q4_0'),
           ('group_q4_1','Q4_1','q4_1')]
    baseline_ppl=None; baseline_pred=None
    for name,qtype,head in cases:
        print('GGUF START',args.recipient,name,flush=True)
        start=time.perf_counter()
        try:
            path=f32 if qtype is None else out/f'{name}.gguf'
            if qtype:
                qargs=[args.llama_root/'build/bin/llama-quantize','--pure','--token-embedding-type','f32']
                if head=='keep': qargs+=['--leave-output-tensor']
                else:qargs+=['--output-tensor-type',head]
                command(qargs+[f32,path,qtype,str(args.threads)],out/f'{name}.log')
            info=inspect_gguf(path)
            types=info['selected_tensor_types']
            if qtype and types.get('blk.0.attn_q.weight') != qtype:
                raise ValueError(f'backbone type unexpected: {types}')
            if head=='keep' and types.get('output.weight') != 'F32':
                raise ValueError(f'head not kept float: {types}')
            if head not in (None,'keep') and types.get('output.weight') != qtype:
                raise ValueError(f'head not quantized as expected: {types}')
            trace,quality,pred=evaluate_gguf(path,tok,src,windows[384:400],args.threads)
            if baseline_ppl is None:
                baseline_ppl=quality['perplexity'];baseline_pred=pred
            quality['relative_ppl_pct']=100*(quality['perplexity']/baseline_ppl-1)
            quality['agreement_vs_native_f32']=sum(sum(a==b for a,b in zip(p,q)) for p,q in zip(pred,baseline_pred))/quality['tokens']
            quality['demo_utility_pass']=(quality['relative_ppl_pct']<=10 and quality['agreement_vs_native_f32']>=.9)
            result['branches'][name]={'trace':trace,'quality':quality,'storage':info,
                                      'elapsed_seconds':time.perf_counter()-start}
            dump(out/'metrics.json',result)
            print('GGUF RESULT',json.dumps({'user':args.recipient,'name':name,'trace':trace,'quality':quality},ensure_ascii=False),flush=True)
        except Exception as exc:
            result['errors'][name]=f'{type(exc).__name__}: {exc}'
            dump(out/'metrics.json',result)
            print('GGUF ERROR',args.recipient,name,type(exc).__name__,str(exc),flush=True)
            if name=='baseline_f32':raise
    result['status']='completed' if not result['errors'] else 'completed_with_errors'
    dump(out/'metrics.json',result)
    print('GGUF COMPLETE',args.recipient,result['status'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipient',type=int,choices=(1,2,3),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--llama-root',type=Path,default=Path('/workspace/tvmc_tools/llama.cpp'))
    p.add_argument('--run-dir',type=Path,default=Path('work/multiprocess_real/run2'))
    p.add_argument('--corpus',type=Path,default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'))
    p.add_argument('--threads',type=int,default=1)
    run(p.parse_args())

if __name__=='__main__':main()
