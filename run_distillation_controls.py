"""Additional attacks on saved benchmark adapters; never retrains the defender.

Keeps the short-budget soft attack and a utility-regularized hard-label attacker
so a 500-step probability-distillation success cannot hide easy removal attacks.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime,timezone
from uuid import uuid4
from run_demo import Tee


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--model-path',type=Path)
    p.add_argument('--offline',action='store_true')
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    primary=json.loads((args.run_dir/'metrics.json').read_text(encoding='utf-8'))
    cache=Path(__file__).resolve().parent/'.cache'/'huggingface'
    os.environ.setdefault('HF_HOME',str(cache))
    out=args.run_dir/'distillation_controls'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:6])
    out.mkdir(parents=True)
    result={'parent_run':primary['run_id'],'status':'running','attacks':{},
            'helper_sha256':hashlib.sha256((Path(__file__).resolve().parent/'tvmc'/'robustness.py').read_bytes()).hexdigest()}
    def save(): (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    code=3
    with (out/'demo_output.log').open('w',encoding='utf-8') as log:
      with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        try:
            import numpy as np
            import torch
            from huggingface_hub import snapshot_download
            from tvmc.model import ModelLab,MODEL_ID
            from tvmc.tardos import Tardos
            from tvmc.robustness import corpus,utility,utility_result,natural_triggers,training_cache,student_distillation,margins,aggregate,MODEL_REVISION
            source=args.model_path or Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(cache),local_files_only=args.offline,allow_patterns=['*.json','model.safetensors','merges.txt','vocab.json']))
            lab=ModelLab(str(source),cache,'cpu',args.threads,True)
            data=corpus(cache,args.offline,primary['dataset'].get('test_offset',0))
            hashes=[hashlib.sha256(x.encode()).hexdigest() for x in data['test']]
            if hashes!=primary['dataset']['hashes']['test']: raise ValueError('test split mismatch')
            result['test_hashes']=hashes
            base,preds=utility(lab,data['test']); result['clean_test']=base
            clean_state=lab.snapshot()
            cache_values=training_cache(lab,data['distillation'])
            t=Tardos(); words=np.asarray(json.loads((args.run_dir/'benchmark_codebook.json').read_text())['words'])
            result['clean_student_negative_controls']={}
            for profile in ('original','natural_redundant'):
                lab.restore(clean_state)
                prompts=lab.triggers(t.master,256) if profile=='original' else natural_triggers(lab,data['triggers'])
                r=1 if profile=='original' else 3
                bits=aggregate(margins(lab,prompts),r)
                decoded=t.decode(words,bits)
                row={'decoded_users':[i+1 for i in decoded], 'innocent_accusations':len(decoded),
                     'utility':utility_result(base,base), 'teacher_training_steps':0}
                result['clean_student_negative_controls'][profile]=row
                print('Clean public student without teacher training',profile,json.dumps(row),flush=True)
                save()
            for profile in ('original','natural_redundant'):
                state=torch.load(args.run_dir/f'{profile}_user1.pt',weights_only=True,map_location='cpu')
                prompts=lab.triggers(t.master,256) if profile=='original' else natural_triggers(lab,data['triggers'])
                r=1 if profile=='original' else 3
                for mode,steps in [('soft',12),('hard_utility',500)]:
                    print('Attack',profile,mode,steps,flush=True)
                    lab.restore(state); torch.manual_seed(2026)
                    training=student_distillation(lab,cache_values,mode,steps)
                    bits=aggregate(margins(lab,prompts),r)
                    decoded=t.decode(words,bits)
                    u,_=utility(lab,data['test'],preds)
                    row={'training':training,'decoded_users':[i+1 for i in decoded],
                         'bit_accuracy':float(np.mean(bits==words[0])),'fingerprint_survives':0 in decoded,
                         'innocent_accused':len(set(decoded)-{0}), 'utility':utility_result(u,base)}
                    result['attacks'][f'{profile}/{mode}/{steps}']=row
                    print(json.dumps(row),flush=True); save()
            result['status']='completed'; code=0
        except Exception as exc:
            traceback.print_exc(); result.update(status='error',error=f'{type(exc).__name__}: {exc}')
        finally:
            save(); print('Controls saved:',out,flush=True)
    return code


if __name__=='__main__': raise SystemExit(main())
