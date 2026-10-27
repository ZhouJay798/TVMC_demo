"""Five fresh paired repetitions; fixed strength, no candidate selection."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch
from tvmc.model import ModelLab
from tvmc.robustness import corpus
from tvmc.inheritance import FullHeadLoRA, cache_features, evaluate, detect, key_vector, train
from tvmc.text_watermark import generate, served_utility


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--offline',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent;cache=root/'.cache/huggingface'
    m={'status':'running','protocol':{'pairs':5,'generation_seeds':list(range(4502,4507)),
        'student_seeds':list(range(3302,3307)),'strengths':[0.,.2],
        'steps':500,'clean_kl':10,'generation_documents':64,'tokens_per_document':32,
        'test_offset':256,'note':'fixed existing test set and user 1; independent fresh seed pairs; no retuning; prior 4501/3301 trial excluded'},
        'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['run_text_repeats.py','tvmc/text_watermark.py','tvmc/inheritance.py']},
        'teacher_sha256':hashlib.sha256(args.teacher.read_bytes()).hexdigest(),'rows':[]}
    def save():(args.output/'metrics.json').write_text(json.dumps(m,indent=2)+'\n')
    save()
    lab=ModelLab(str(args.model_path),cache,'cpu',4,True)
    lab.head=FullHeadLoRA(lab.head.base,8);lab.model.lm_head=lab.head
    data=corpus(cache,args.offline,256)
    m['data_hashes']={k:[hashlib.sha256(t.encode()).hexdigest() for t in v] for k,v in data.items()}
    test=cache_features(lab,data['test'],96)
    m['clean'],_=evaluate(lab,test);base=m['clean']['perplexity']
    teacher=torch.load(args.teacher,map_location='cpu',weights_only=True)
    seeds=[60101,60202,60303,60404,60505]
    key=torch.tensor(key_vector(lab.head.base.out_features,seeds[0]))
    lab.head.restore(teacher)
    m['teacher']={str(s):served_utility(lab,test,key,s) for s in [0.,.2]}
    for r in m['teacher'].values():r['increase']=r['perplexity']/base-1
    save()
    for index,(gs,ss) in enumerate(zip(range(4502,4507),range(3302,3307)),1):
        for strength in [0.,.2]:
            print('Pair',index,'strength',strength,'generation seed',gs,'student seed',ss,flush=True)
            lab.head.restore(teacher)
            generated,records=generate(lab,data['distillation'],key,strength,seed=gs)
            (args.output/f'text_pair{index}_{strength}.json').write_text(json.dumps(records))
            training=train(lab,generated,'generated_utility',500,clean_kl=10,seed=ss)
            u,d=evaluate(lab,test);detection=detect(d,seeds)
            row={'pair':index,'strength':strength,'generation_seed':gs,'student_seed':ss,
                 'training':training,'utility':u,'increase':u['perplexity']/base-1,
                 'detection':detection,'success':detection['decoded_users']==[1]}
            m['rows'].append(row)
            torch.save(lab.head.snapshot(),args.output/f'student_pair{index}_{strength}.pt')
            save();print(json.dumps(row),flush=True)
    m['summary']={}
    for strength in [0.,.2]:
        rows=[r for r in m['rows'] if r['strength']==strength]
        vals=[r['utility']['perplexity'] for r in rows]
        m['summary'][str(strength)]={'successes':sum(r['success'] for r in rows),'trials':len(rows),
          'ppl_mean':statistics.mean(vals),'ppl_sample_sd':statistics.stdev(vals),'ppl_min':min(vals),'ppl_max':max(vals),
          'increase_mean':statistics.mean(r['increase'] for r in rows),
          'false_accusation_runs':sum(any(u!=1 for u in r['detection']['decoded_users']) for r in rows)}
    m['status']='completed';save();print(json.dumps(m['summary']),flush=True)

if __name__=='__main__':main()
