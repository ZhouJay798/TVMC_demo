"""Measure held-out PPL of the already-tested 50/50 effective head mixtures."""
import argparse,json,itertools
from pathlib import Path
import torch
from tvmc.model import ModelLab,HeadLoRA
from tvmc.robustness import corpus,utility


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();out=a.run_dir
    metrics=json.loads((out/'model_metrics.json').read_text());assert metrics['status']=='completed'
    root=Path(__file__).resolve().parent;cache=root/'.cache/huggingface';lab=ModelLab(str(root/'models/SmolLM2-135M'),cache,'cpu',2,True)
    data=corpus(cache,True,256);baseline,preds=utility(lab,data['test']);states=[torch.load(out/(f'C-E0-U{i}.pt'),map_location='cpu',weights_only=True) for i in (1,2,3)]
    rows=[]
    for i,j in itertools.combinations(range(3),2):
        head=HeadLoRA(lab.head.base,lab.token_ids,rank=16)
        with torch.no_grad():head.a.copy_(torch.cat([states[i]['a'],states[j]['a']],0));head.b.copy_(torch.cat([states[i]['b'],states[j]['b']],1))
        lab.head=head;lab.model.lm_head=head;u,_=utility(lab,data['test'],preds)
        rows.append({'pair':[f'C-E0-U{i+1}',f'C-E0-U{j+1}'],'ppl':u['perplexity'],
                     'ppl_increase_pct':100*(u['perplexity']/baseline['perplexity']-1),'next_token_agreement':u['next_token_agreement']})
        print(json.dumps(rows[-1]),flush=True)
    (out/'mixture_utility.json').write_text(json.dumps({'baseline':baseline,'rows':rows},indent=2)+'\n')

if __name__=='__main__':main()
