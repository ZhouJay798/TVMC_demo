"""Fixed natural teachers; select global generation bias on validation, test all users."""
import argparse,json
from pathlib import Path
import torch
from tvmc.model import ModelLab
from tvmc.robustness import corpus
from tvmc.inheritance import FullHeadLoRA,cache_features,evaluate,detect,key_vector,train
from tvmc.text_watermark import generate,served_utility
from run_integrated_trace import recache_student


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model-path',type=Path,required=True);p.add_argument('--natural-dir',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--offline',action='store_true')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cache=Path(__file__).resolve().parent/'.cache/huggingface'
 lab=ModelLab(str(a.model_path),cache,'cpu',4,True);basehead=lab.head.base;ids=lab.token_ids
 lab.head=FullHeadLoRA(basehead,8);lab.model.lm_head=lab.head
 data=corpus(cache,a.offline,384);val=cache_features(lab,data['validation'],96);test=cache_features(lab,data['test'],96)
 cleanval,_=evaluate(lab,val);cleantest,_=evaluate(lab,test)
 seeds=[60101,60202,60303,60404,60505];states=[]
 for i in range(1,6):
  s=torch.load(a.natural_dir/f'natural_user{i}.pt',weights_only=True);b=torch.zeros_like(lab.head.b);b[ids]=s['b'];states.append({'a':s['a'],'b':b})
 m={'protocol':{'bias_candidates':[.5,1.],'generation_seed':4501,'student_seed':3301,'tokens':2048,'steps':500,'kl':10,'test_offset':384,'selection':'first strength with all five validation users correctly located; no test tuning','scope':'natural teacher plus generation-layer bias; no dense teacher adapter'},'clean_validation':cleanval,'clean_test':cleantest,'development':[],'final':[]}
 def save():(a.output/'metrics.json').write_text(json.dumps(m,indent=2)+'\n')
 selected=None
 for strength in [.5,1.]:
  rows=[]
  for i in range(5):
   print('Develop',strength,'user',i+1,flush=True)
   lab.head.restore(states[i]);key=torch.tensor(key_vector(basehead.out_features,seeds[i]))
   teacher=served_utility(lab,val,key,strength)
   _,records=generate(lab,data['distillation'],key,strength)
   (a.output/f'text_{strength}_{i+1}.json').write_text(json.dumps(records))
   c=recache_student(lab,records);train(lab,c,'generated_utility',500,clean_kl=10,seed=3301)
   torch.save(lab.head.snapshot(),a.output/f'student_{strength}_{i+1}.pt')
   u,d=evaluate(lab,val);r={'user':i+1,'strength':strength,'teacher':teacher,'student':u,'detection':detect(d,seeds)}
   rows.append(r);m['development'].append(r);save();print(r,flush=True)
  if all(r['detection']['decoded_users']==[r['user']] for r in rows):selected=strength;break
 m['selected']=selected
 if selected is not None:
  for i in range(5):
   key=torch.tensor(key_vector(basehead.out_features,seeds[i]));lab.head.restore(states[i])
   teacher=served_utility(lab,test,key,selected)
   lab.head.restore(torch.load(a.output/f'student_{selected}_{i+1}.pt',weights_only=True))
   u,d=evaluate(lab,test)
   r={'user':i+1,'teacher':teacher,'teacher_increase':teacher['perplexity']/cleantest['perplexity']-1,'student':u,'student_increase':u['perplexity']/cleantest['perplexity']-1,'detection':detect(d,seeds)}
   m['final'].append(r);save();print('Final',r,flush=True)
 lab.head.reset();u,d=evaluate(lab,test);m['clean_negative']=detect(d,seeds)
 m['all_users_located']=len(m['final'])==5 and all(r['detection']['decoded_users']==[r['user']] for r in m['final'])
 m['status']='completed';save();print('Completed',m['all_users_located'],flush=True)
if __name__=='__main__':main()
