"""Compare a protected head with its exact public base and prune changed rows.

Attack selection uses only the two model heads and a tolerance, not the secret
detector key. The private key is used later solely to verify the result.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file

from run_finetune_fingerprint_experiment import _extract_model_archive, corpus_windows, dump, load_local, verify_source
from run_prune_resilient_experiment import get_trace, prompt_features, quality, secret_pairs


def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipient',type=int,choices=(1,2,3),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--key',type=Path,required=True)
    p.add_argument('--prototype',type=Path,default=Path('work/prune_resilience/run1'))
    p.add_argument('--rows-file',default='experimental_head_rows.safetensors')
    p.add_argument('--rows-key',default='prune_aware_rows')
    p.add_argument('--run-dir',type=Path,default=Path('work/multiprocess_real/run2'))
    p.add_argument('--corpus',type=Path,default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'))
    p.add_argument('--clean-base',type=Path,default=Path('models/SmolLM2-135M-Instruct-Q4_1-converted'))
    p.add_argument('--cache',type=Path,default=Path('.cache/huggingface'))
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--tolerance',type=float,default=1e-6)
    a=p.parse_args();torch.set_num_threads(a.threads)
    src=verify_source(a.run_dir.resolve(),a.recipient)
    base_path=a.clean_base/'model.safetensors'
    digest=file_sha256(base_path)
    if digest!=src['source_metrics']['base_sha256']:
        raise ValueError('clean base differs from registered source')
    windows,corpus_hash=corpus_windows(a.corpus)
    if corpus_hash!=src['source_metrics']['text_corpus_sha256']:raise ValueError('corpus mismatch')
    a.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='diff-source-',dir=a.output) as tmp:
        source_dir=_extract_model_archive(src['payload'],tmp)
        model,tok,legacy=load_local(source_dir,a.cache,a.threads)
        base,_,_=load_local(a.clean_base,a.cache,a.threads)
        rows=load_file(str(a.prototype/f'U{a.recipient}'/a.rows_file))
        alt=rows['token_ids'].tolist()
        with torch.no_grad():model.lm_head.weight[alt]=rows[a.rows_key]
        # The attacker's rule is purely a head-to-public-base weight comparison;
        # it does not use `legacy`, `alt`, `slots`, probes, codeword, or key.
        with torch.no_grad():
            diffs=torch.linalg.vector_norm(model.lm_head.weight-base.lm_head.weight,dim=1)
            selected=(diffs>a.tolerance).nonzero(as_tuple=True)[0]
        if len(selected)==0:raise ValueError('difference rule found no rows')
        key=a.key.read_bytes()
        expected_alt,slots=secret_pairs(key,model.lm_head.weight.shape[0],set(tok.all_special_ids)|set(legacy))
        if alt!=expected_alt:raise ValueError('local verifier key and experimental head mismatch')
        h=prompt_features(model,tok,src['prompts'])
        before=get_trace(h,model.lm_head.weight,legacy,alt,slots,src)
        qb,preds=quality(model,tok,windows[384:400])
        with torch.no_grad():model.lm_head.weight[selected]=0
        after=get_trace(h,model.lm_head.weight,legacy,alt,slots,src)
        qa,_=quality(model,tok,windows[384:400],preds)
        qa['relative_ppl_pct_vs_before']=100*(qa['perplexity']/qb['perplexity']-1)
        selected_set=set(selected.tolist())
        out={'recipient':a.recipient,'status':'completed','source_sha256':src['payload_sha256'],
             'base_sha256':digest,'attacker_assumptions':'exact clean base + protected model head; no detector key or probes for row selection',
             'tolerance':a.tolerance,'head_rows_selected':len(selected_set),
             'total_head_rows':model.lm_head.weight.shape[0],
             'known_primary_rows_selected':len(selected_set.intersection(legacy)),
             'known_alternate_rows_selected':len(selected_set.intersection(alt)),
             'before':{'trace':before,'quality':qb},'after':{'trace':after,'quality':qa},
             'limits':'controlled model with exact public base and FP32 head; does not prove identical effectiveness for other architectures'}
        path=a.output/f'U{a.recipient}.json';dump(path,out)
        print(json.dumps({'user':a.recipient,'selected_rows':len(selected_set),
              'before_hits':before['combined_hits'],'after_hits':after['combined_hits'],
              'before_ppl':qb['perplexity'],'after_ppl':qa['perplexity'],
              'relative_pct':qa['relative_ppl_pct_vs_before']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
