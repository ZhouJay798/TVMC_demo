"""Matched 30%-sparse recovery controls on saved research adapters.

The teacher is the public clean base. Neither secret triggers nor code labels
enter recovery. Existing zeros are frozen, so recovery cannot undo sparsity.
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
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    primary=json.loads((args.run_dir/'metrics.json').read_text(encoding='utf-8'))
    cache=Path(__file__).resolve().parent/'.cache'/'huggingface'
    os.environ.setdefault('HF_HOME',str(cache))
    out=args.run_dir/'pruning_recovery'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:6])
    out.mkdir(parents=True)
    result={'parent_run':primary['run_id'],'status':'running','methods':{},
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
            from tvmc.robustness import (corpus,utility,utility_result,natural_triggers,
                margins,aggregate,MODEL_REVISION,hessians,compressed,recovery_teacher_logits,recover_sparse)
            source=args.model_path or Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(cache),local_files_only=args.offline,allow_patterns=['*.json','model.safetensors','merges.txt','vocab.json']))
            if hashlib.sha256((source/'model.safetensors').read_bytes()).hexdigest()!=primary['model']['sha256']:
                raise ValueError('different base model from primary experiment')
            lab=ModelLab(str(source),cache,'cpu',args.threads,True)
            clean=lab.snapshot()
            data=corpus(cache,args.offline,primary['dataset']['test_offset'])
            for split in ('test','calibration'):
                hashes=[hashlib.sha256(x.encode()).hexdigest() for x in data[split]]
                if hashes!=primary['dataset']['hashes'][split]: raise ValueError(f'{split} split mismatch')
            base,preds=utility(lab,data['test']); result['clean_test']=base
            print('Calibrating fixed masks and caching clean teacher',flush=True)
            grams=hessians(lab,data['calibration'])
            teacher=recovery_teacher_logits(lab,data['calibration'],50)
            t=Tardos(); words=np.asarray(json.loads((args.run_dir/'benchmark_codebook.json').read_text())['words'])
            for method in ('magnitude30','sparsegpt30'):
                lab.restore(clean)
                with compressed(lab,method,grams) as compression:
                    print('Recovering',method,flush=True)
                    training=recover_sparse(lab,data['calibration'],50,3e-5,teacher)
                    u,_=utility(lab,data['test'],preds)
                    row={'compression':compression,'training':training,'clean_control':utility_result(u,base)}
                    for profile in ('original','natural_redundant'):
                        state=torch.load(args.run_dir/f'{profile}_user1.pt',weights_only=True,map_location='cpu')
                        lab.restore(state)
                        prompts=lab.triggers(t.master,256) if profile=='original' else natural_triggers(lab,data['triggers'])
                        r=1 if profile=='original' else 3
                        bits=aggregate(margins(lab,prompts),r); decoded=t.decode(words,bits)
                        u,_=utility(lab,data['test'],preds)
                        row[profile]={'decoded_users':[i+1 for i in decoded],
                            'bit_accuracy':float(np.mean(bits==words[0])),'fingerprint_survives':0 in decoded,
                            'innocent_accused':len(set(decoded)-{0}),'utility':utility_result(u,base),
                            'legacy_utility':utility_result(lab.utility(),lab.clean_utility)}
                    result['methods'][method]=row
                    print(json.dumps(row),flush=True); save()
            result['status']='completed'; code=0
        except Exception as exc:
            traceback.print_exc(); result.update(status='error',error=f'{type(exc).__name__}: {exc}')
        finally:
            save(); print('Recovery controls saved:',out,flush=True)
    return code


if __name__=='__main__': raise SystemExit(main())
