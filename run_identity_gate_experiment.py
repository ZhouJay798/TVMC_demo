"""Five registered identities, >=2 admission, three real model deliveries.

Only repetition 1 is used to train the three models. Five cryptographic
repetitions are independent but do not represent five model trainings.
The registry is a trusted local prototype; real-person enrollment and private
transport are not provided.
"""
import argparse,copy,hashlib,io,itertools,json,secrets,tempfile,threading,urllib.request,urllib.error,zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tvmc.crypto import G,P,Q,Share,scalar,prove,recover,encrypt,decrypt,canonical
from tvmc.epochs import EpochPublic
from tvmc.identity_gate import IdentityGate,enrollment_context,identity_proof
from tvmc.tardos import Tardos
from tvmc.model import ModelLab
from tvmc.robustness import corpus,natural_triggers,training_cache,embed_natural,margins,aggregate,utility
from serve import make_server

SCENARIOS=[('two',[1,2]),('three',[2,3,4]),('five',[1,2,3,4,5])]

def sha(data):return hashlib.sha256(data).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def rejected(fn):
    try:fn();return False
    except (ValueError,KeyError):return True

def enrolled():
    secrets_=[scalar() for _ in range(5)]
    records=[{'i':i,'commitment':pow(G,secrets_[i-1],P),
              'proof':identity_proof(i,secrets_[i-1],enrollment_context(i))} for i in range(1,6)]
    return secrets_,records,IdentityGate(records)

def new_session(gate,secrets_,sid,ids,basehash):
    nonce=secrets.token_hex(32);ctx=gate.context(sid,0,ids,basehash,nonce)
    proofs=[identity_proof(i,secrets_[i-1],ctx) for i in ids]
    return gate.register(sid,0,ids,basehash,nonce,proofs),ctx,proofs

def key_material(gate,secrets_,admission,threshold):
    gate.require_admitted(admission)
    ids=admission.participants
    if threshold!=len(ids):raise ValueError('all registered participants required')
    coefficients=[scalar() for _ in range(threshold)];key=coefficients[0]
    public=EpochPublic(tuple(pow(G,a,P) for a in coefficients),tuple(gate.registry[i] for i in range(1,6)),5,threshold,ids)
    shares=[Share(i,sum(a*pow(i,j,Q) for j,a in enumerate(coefficients))%Q,secrets_[i-1]) for i in ids]
    context={'domain':'TVMC/admitted-key-recovery/v1','sid':admission.sid,'epoch':admission.epoch,
             'participants':list(ids),'threshold':threshold,'seed':admission.seed,'base_sha256':admission.base_sha256,'nonce':admission.nonce}
    proofs=[prove(public,s,context) for s in shares]
    return key,public,shares,proofs,context

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--threshold-policy',choices=['all'],default='all');args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent;source=root/'models/SmolLM2-135M';cache=root/'.cache/huggingface'
    basehash=sha((source/'model.safetensors').read_bytes());crypto=[];private=[];public_repeats=[]
    for repeat in range(1,6):
        identities,enrollments,gate=enrolled();admissions=[]
        # Invalid requests are attempted before admission; no registered session is allowed to appear.
        ids=[1,2];ctx=gate.context('negative',0,ids,basehash,'challenge');p=[identity_proof(i,identities[i-1],ctx) for i in ids]
        before=len(gate.sessions)
        badp=copy.deepcopy(p);badp[0]['z']=(badp[0]['z']+1)%Q
        cases={'single':lambda:gate.register('negative',0,[1],basehash,'challenge',p[:1]),
          'duplicate':lambda:gate.register('negative',0,[1,1],basehash,'challenge',[p[0],p[0]]),
          'missing':lambda:gate.register('negative',0,ids,basehash,'challenge',p[:1]),
          'tampered':lambda:gate.register('negative',0,ids,basehash,'challenge',badp),
          'cross_context_replay':lambda:gate.register('other',0,ids,basehash,'challenge',p),
          'unknown_custodian':lambda:gate.register('negative',0,[1,6],basehash,'challenge',p)}
        negatives={name:rejected(fn) for name,fn in cases.items()};negatives['no_session_created']=len(gate.sessions)==before
        for label,ids in SCENARIOS:
            admission,ctx,proofs=new_session(gate,identities,'session-'+label,ids,basehash)
            key,public,shares,keyproofs,keyctx=key_material(gate,identities,admission,len(ids))
            subsets=[]
            for indices in itertools.combinations(range(len(ids)),public.threshold):
                result=recover(public,[shares[i] for i in indices],[keyproofs[i] for i in indices],keyctx)
                subsets.append(result==key)
            ciphertext=encrypt(key,b'admission test payload',keyctx)
            checks={'all_authorized_subsets_recover':all(subsets),'authorized_subset_count':len(subsets),
                    'all_participants_received_one_share':len(shares)==len(ids) and {s.i for s in shares}==set(ids),
                    'below_threshold_rejected':rejected(lambda:recover(public,shares[:public.threshold-1],keyproofs[:public.threshold-1],keyctx)),
                    'duplicate_rejected':rejected(lambda:recover(public,[shares[0]]*public.threshold,[keyproofs[0]]*public.threshold,keyctx)),
                    'wrong_session_proof_rejected':rejected(lambda:recover(public,shares[:public.threshold],keyproofs[:public.threshold],dict(keyctx,sid='other'))),
                    'decrypt':decrypt(key,ciphertext,keyctx)==b'admission test payload'}
            crypto.append({'repeat':repeat,'label':label,'participants':ids,'admission':True,'negatives':negatives,'checks':checks})
            admissions.append({'admission':asdict(admission),'request_context':ctx,'request_proofs':proofs,'key_public':asdict(public),'key_proofs':keyproofs,'key_context':keyctx})
            if repeat==1:private.append((admission,key,public,shares,keyproofs,keyctx,gate))
        public_repeats.append({'repeat':repeat,'enrollment':enrollments,'admissions':admissions})
        print('CRYPTO REPEAT',repeat,'passed',all(negatives.values()) and all(all(v for k,v in x['checks'].items() if k!='authorized_subset_count') for x in crypto if x['repeat']==repeat),flush=True)
    dump(out/'crypto_metrics.json',{'repetitions':5,'threshold_policy':args.threshold_policy,'rows':crypto,'public_repeats':public_repeats})
    t=Tardos(n=3);words=[t.encode(x[0].seed) for x in private]
    (out/'registration_before_training.json').write_text(json.dumps({'records':public_repeats[0]['admissions'],'p':t.p.tolist(),'threshold':t.threshold},indent=2))
    lab=ModelLab(str(source),cache,'cpu',4,True);data=corpus(cache,True,256);prompts=natural_triggers(lab,data['triggers'],3)
    print('Preparing shared fixed features and baseline',flush=True)
    base,preds=utility(lab,data['test']);features=lab.features(prompts);cal=training_cache(lab,data['calibration']);rows=[]
    def query_model(active):
        server=make_server(lambda ps:margins(active,ps),port=0,max_prompts=768,response_field='score_differences')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_address[1]}/query';body=canonical({'protocol':'batch','prompts':prompts})
        client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def send():return client.open(urllib.request.Request(url,body,{'Content-Type':'application/json'}),timeout=600)
        try:
            with send() as r:first=r.status;response=json.load(r)
            try:
                with send() as r:second=r.status
            except urllib.error.HTTPError as e:second=e.code
        finally:server.shutdown();server.server_close();thread.join()
        observed=aggregate(response['score_differences'],3);decoded=t.decode(words,observed)
        return first,second,response,observed,decoded
    first,second,response,observed,decoded=query_model(lab)
    dump(out/'clean_negative.json',{'decoded_sessions':[private[i][0].sid for i in decoded],'first_http':first,'second_http':second,'score_differences':response['score_differences']})
    for index,(admission,key,public,shares,keyproofs,keyctx,session_gate) in enumerate(private):
        session_gate.require_admitted(admission)
        sid=admission.sid;torch.manual_seed(81101+index);print('TRAIN',sid,flush=True)
        train=embed_natural(lab,features,np.repeat(words[index],3),cal)
        adapter=out/(sid+'.pt');torch.save(lab.snapshot(),adapter)
        archive=out/(sid+'.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
            for path in sorted(source.iterdir()):
                if path.is_file():z.write(path,'base/'+path.name)
            z.write(adapter,'adapter.pt');z.writestr('registration.json',json.dumps(public_repeats[0]['admissions'][index]))
        payload=archive.read_bytes();payloadhash=sha(payload);aad={'sid':sid,'payload_sha256':payloadhash,'admission_seed':admission.seed}
        session_gate.require_admitted(admission)
        ciphertext=encrypt(key,payload,aad);(out/(sid+'.aesgcm')).write_bytes(ciphertext)
        session_gate.require_admitted(admission)
        restored=recover(public,shares[:public.threshold],keyproofs[:public.threshold],keyctx);plaintext=decrypt(restored,ciphertext,aad)
        hashmatch=sha(plaintext)==payloadhash;del payload
        with tempfile.TemporaryDirectory(dir=out) as temp:
            temp=Path(temp)
            with zipfile.ZipFile(io.BytesIO(plaintext)) as z:z.extractall(temp)
            del plaintext
            active=ModelLab(str(temp/'base'),cache,'cpu',2,True);active.restore(torch.load(temp/'adapter.pt',weights_only=True,map_location='cpu'))
            first,second,response,observed,decoded=query_model(active);u,_=utility(active,data['test'],preds)
            row={'sid':sid,'participants':admission.participants,'admitted_proofs':len(admission.proofs),
                'recovery_threshold':f'{public.threshold}-of-{len(shares)}','recovery_participants':[s.i for s in shares[:public.threshold]],'distributed_share_count':len(shares),
                'recovered_key_matches':restored==key,'decrypted_hash_matches':hashmatch,'decrypted_model_reloaded':True,
                'model_sha256':payloadhash,'cipher_sha256':sha(ciphertext),'decoded_sessions':[private[i][0].sid for i in decoded],
                'correct_localization':decoded==[index],'observed_bits':int((observed>=0).sum()),'correct_bits':int((observed==words[index]).sum()),
                'wrong_session_hits':[private[i][0].sid for i in decoded if i!=index],
                'ppl':u['perplexity'],'ppl_increase_pct':100*(u['perplexity']/base['perplexity']-1),
                'first_http':first,'second_http':second,'training':train}
            rows.append(row);dump(out/'model_metrics.json',{'status':'running','baseline':base,'rows':rows})
            dump(out/(sid+'_http.json'),{'input_count':768,'score_differences':response['score_differences'],'observed':observed.tolist()})
            print('MODEL RESULT',json.dumps(row),flush=True);del active
    dump(out/'model_metrics.json',{'status':'completed','baseline':base,'rows':rows})
    print('COMPLETE',len(rows),'models',flush=True)

if __name__=='__main__':main()
