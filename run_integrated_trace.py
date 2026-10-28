"""Registered natural fingerprints, calibrated pruning, and sampled-text inheritance.
Both paper-style natural-code evidence and separate research statistical receipts
are exported, including failures. No replacement of original entry points.
"""
import argparse,json,hashlib,copy
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
from tvmc.experiment import core_experiment
from tvmc.model import ModelLab
from tvmc.robustness import corpus,natural_triggers,training_cache,embed_natural,margins,aggregate,hessians,compressed
from tvmc.inheritance import FullHeadLoRA,cache_features,evaluate,key_vector,train
from tvmc.text_watermark import generate,served_utility
from tvmc.bridge import Collector,OneShotBox,evidence_for
from tvmc.crypto import prove,recover,encrypt,decrypt,canonical,digest
from verify_evidence import verify
from tvmc.statistical_judge import verify_statistical


@torch.no_grad()
def recache_student(lab,records):
    """Recompute clean-student features: pruned teacher features cannot be reused."""
    hs=[];ys=[]
    for start in range(0,len(records),4):
        rows=records[start:start+4]
        ids=torch.tensor([r['prompt_token_ids']+r['completion_token_ids'] for r in rows],device=lab.device)
        h=lab.model.model(input_ids=ids,use_cache=False).last_hidden_state[:,15:-1]
        hs.append(h.reshape(-1,h.shape[-1]).cpu());ys.append(ids[:,16:].reshape(-1).cpu())
    return {'h':torch.cat(hs),'labels':torch.cat(ys),'documents':len(records)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--offline',action='store_true')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent;cache=root/'.cache/huggingface'
    core,state=core_experiment(trials=2000)
    source=args.model_path
    raw=(source/'model.safetensors').read_bytes();sha=hashlib.sha256(raw).hexdigest()
    aad={'asset_sha256':sha,'purpose':'integrated-five-session-release'}
    encrypted=encrypt(state['key'],raw,aad);(out/'model.aesgcm').write_bytes(encrypted)
    recovered_sessions=[]
    for rec in state['records']:
        kc={'sid':rec.sid,'epoch':rec.epoch,'sha256':sha}
        shares=[state['shares'][i-1] for i in rec.participants]
        proofs=[prove(state['public'],s,kc) for s in shares]
        key=recover(state['public'],shares,proofs,kc)
        recovered_sessions.append({'user':rec.consumer,'participants':rec.participants,
            'decrypted_sha256_matches':hashlib.sha256(decrypt(key,encrypted,aad)).hexdigest()==sha})
    del raw,encrypted
    lab=ModelLab(str(source),cache,'cpu',4,True)
    data=corpus(cache,args.offline,256)
    m={'status':'running','core':core,'recovery':recovered_sessions,'asset_sha256':sha,'rows':[],
       'protocol':{'fingerprint':'registered Tardos, natural language, 3 inputs/bit, 256 bits',
       'pruning':'SparseGPT-style second-order compensation 30% backbone Linear weights',
       'text_bias':.2,'generated_tokens':2048,'student_steps':500,'clean_kl':10,
       'generation_seed':4501,'student_seed':3301,'test_offset':256,
       'important':'Natural-only teachers + optional generation bias; no dense statistical teacher added. Different teachers from prior text repeats.'},
       'data_hashes':{k:[hashlib.sha256(t.encode()).hexdigest() for t in v] for k,v in data.items()}}
    def save():(out/'metrics.json').write_text(json.dumps(m,indent=2)+'\n')
    test=cache_features(lab,data['test'],96);m['clean'],_=evaluate_proxy(lab,test)
    base=m['clean']['perplexity'];prompts=natural_triggers(lab,data['triggers'],3)
    features=lab.features(prompts);cal=training_cache(lab,data['calibration'])
    collector=Collector();seeds=[60101,60202,60303,60404,60505]
    contexts=[]
    for rec in state['records']:
        kc={'sid':rec.sid,'epoch':rec.epoch,'sha256':sha}
        contexts.append({'public':asdict(state['public']),'records':[asdict(r) for r in state['records']],
          'tardos_p':state['tardos'].p.tolist(),'threshold':state['tardos'].threshold,
          'collector_key':collector.public_key.public_bytes(Encoding.Raw,PublicFormat.Raw).hex(),
          'now':state['now'],'key_context':kc,
          'key_proofs':[prove(state['public'],state['shares'][i-1],kc) for i in rec.participants],
          'statistical_key_seeds':seeds,'statistical_vocab':lab.head.base.out_features})
    (out/'registration_before_training.json').write_text(json.dumps(contexts,indent=2))
    states=[]
    for i in range(5):
        print('Embed registered natural user',i+1,flush=True)
        embed_natural(lab,features,np.repeat(state['words'][i],3),cal)
        states.append(lab.snapshot());torch.save(states[-1],out/f'natural_user{i+1}.pt')
    original_head=lab.head
    lab.head=FullHeadLoRA(original_head.base,8);lab.model.lm_head=lab.head
    fullstates=[]
    for s in states:
        b=torch.zeros_like(lab.head.b);b[lab.token_ids]=s['b']
        fullstates.append({'a':s['a'],'b':b})
    lab.head.reset();grams=hessians(lab,data['calibration'])
    def measure(i,stage,served=False):
        rec=state['records'][i];key=torch.tensor(key_vector(lab.head.base.out_features,seeds[i]))
        # Fresh features under current backbone; never reuse clean features for pruned PPL.
        current=cache_features(lab,data['test'],96)
        u,drift=evaluate(lab,current)
        context=dict(contexts[i],endpoint=f'user{i+1}/{stage}',nonce=f'challenge/{i+1}/{stage}')
        receipt,sig=collector.collect(OneShotBox(lambda ps:aggregate(margins(lab,ps),3)),prompts,'batch',context['endpoint'],context['nonce'])
        ev=evidence_for(rec,state['words'][i],state['openings'][i],receipt,sig)
        natural=verify(context,ev)
        stats_receipt={'endpoint':context['endpoint'],'nonce':context['nonce'],'context_sha256':digest(context),'drift':drift.tolist()}
        stats={'receipt':stats_receipt,'signature':collector._key.sign(canonical(stats_receipt)).hex()}
        statistical=verify_statistical(context,ev,stats)
        decoded=[j+1 for j in state['tardos'].decode(state['words'],receipt['observed'])]
        row={'user':i+1,'session':rec.sid,'participants':rec.participants,'stage':stage,
           'natural_decoded_users':decoded,'natural_verification':natural,'statistical_verification':statistical,
           'utility':u,'ppl_increase':u['perplexity']/base-1}
        if served:
            row['served_utility']=served_utility(lab,current,key,.2)
            row['served_ppl_increase']=row['served_utility']['perplexity']/base-1
        folder=out/f'user_{i+1}'/stage;folder.mkdir(parents=True)
        for name,value in [('public_context.json',context),('public_evidence.json',ev),('statistical_evidence.json',stats),('verdict.json',row)]:
            (folder/name).write_text(json.dumps(value,indent=2)+'\n')
        m['rows'].append(row);save();print('Result',json.dumps(row),flush=True)
    generated_sets={}
    for i in range(5):
        lab.head.restore(fullstates[i]);measure(i,'natural',served=True)
        key=torch.tensor(key_vector(lab.head.base.out_features,seeds[i]))
        _,records=generate(lab,data['distillation'],key,.2)
        generated_sets[i,'text']=records
        (out/f'generated_user{i+1}_text.json').write_text(json.dumps(records))
    print('Applying second order 30% pruning',flush=True)
    with compressed(lab,'sparsegpt30',grams) as info:
        m['pruning_parameters']=info
        for i in range(5):
            lab.head.restore(fullstates[i]);measure(i,'pruned',served=True)
            key=torch.tensor(key_vector(lab.head.base.out_features,seeds[i]))
            _,records=generate(lab,data['distillation'],key,.2)
            generated_sets[i,'pruned_text']=records
            (out/f'generated_user{i+1}_pruned_text.json').write_text(json.dumps(records))
    del grams
    for i in range(5):
        for stage in ['text','pruned_text']:
            print('Student',i+1,stage,flush=True)
            c=recache_student(lab,generated_sets[i,stage])
            train(lab,c,'generated_utility',500,clean_kl=10,seed=3301)
            torch.save(lab.head.snapshot(),out/f'student_user{i+1}_{stage}.pt')
            measure(i,stage)
    lab.head.reset();measure(0,'clean_negative')
    m['status']='completed';m['paper_all_requirements_met']=False
    save()


def evaluate_proxy(lab,cache):
    # evaluate() expects full-vocab head.delta; swap a zero full head for clean base only.
    original=lab.head;lab.head=FullHeadLoRA(original.base,8);lab.model.lm_head=lab.head
    result=evaluate(lab,cache)
    lab.head=original;lab.model.lm_head=original
    return result

if __name__=='__main__':main()
