"""3/3 session, three recipient-bound replicas, real one-shot HTTP tracing.
Five independent crypto fixture runs; three model trainings total.
"""
import argparse,copy,gc,io,itertools,json,tempfile,threading,urllib.request,urllib.error,zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from cryptography.exceptions import InvalidTag,InvalidSignature
from run_identity_gate_experiment import enrolled,new_session,key_material
from tvmc.crypto import canonical,recover,digest
from tvmc.personal_delivery import new_key,bind_recipient,verify_recipient,seal_copy,open_copy,context_for,replica_seed,sha
from tvmc.tardos import Tardos
from tvmc.model import ModelLab
from tvmc.robustness import corpus,natural_triggers,training_cache,embed_natural,margins,aggregate,utility
from serve import make_server


def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def rejects(fn):
    try:fn();return False
    except (ValueError,InvalidTag,InvalidSignature):return True


def setup(basehash):
    ids,enrollments,gate=enrolled();adm,ctx,proofs=new_session(gate,ids,'C-E0',[1,2,3],basehash)
    material=key_material(gate,ids,adm,3);keys=[new_key() for _ in range(3)]
    records=[bind_recipient(gate,adm,i,ids[i-1],keys[i-1]) for i in (1,2,3)]
    return gate,adm,material,keys,records,{'enrollment':enrollments,'admission':asdict(adm),'request_context':ctx,'request_proofs':proofs,
           'key_public':asdict(material[1]),'key_proofs':material[3],'key_context':material[4],'recipients':records}


def crypto_trial(data):
    gate,adm,(k,pub,shares,proofs,ctx),keys,records,_=data
    fixture=b'cryptographic envelope fixture: not model training';packages=[seal_copy(gate,adm,r,k,fixture) for r in records]
    small=[]
    for count in (0,1,2):
        for indices in itertools.combinations(range(3),count):
            small.append(rejects(lambda:recover(pub,[shares[i] for i in indices],[proofs[i] for i in indices],ctx)))
    recovered=recover(pub,shares,proofs,ctx)
    own=[];cross=[];no_session=[];tamper=[];bindings=[]
    for i,p in enumerate(packages):
        own.append(open_copy(recovered,keys[i],p,p['context'])==fixture)
        for j in range(3):
            if i!=j:cross.append(rejects(lambda:open_copy(recovered,keys[j],p,p['context'])))
        no_session.append(rejects(lambda:open_copy(k+1,keys[i],p,p['context'])))
        for field,value in [('sid','other'),('epoch',1),('replica_id','other'),('recipient',(i+1)%3+1)]:
            altered=copy.deepcopy(p);altered['context'][field]=value
            tamper.append(rejects(lambda:open_copy(k,keys[i],altered,altered['context'])))
        for field in ('ciphertext','wrapped_key'):
            altered=copy.deepcopy(p);altered[field]=altered[field][:-1]+bytes([altered[field][-1]^1])
            tamper.append(rejects(lambda:open_copy(k,keys[i],altered,altered['context'])))
        r=copy.deepcopy(records[i]);r['public_key_der']=records[(i+1)%3]['public_key_der']
        bindings.append(rejects(lambda:verify_recipient(gate,adm,r)))
    return {'complete_threshold_recovers':recovered==k,'insufficient_subsets_rejected':small,'owner_decrypts':own,
            'cross_recipient_rejected':cross,'no_session_key_rejected':no_session,'tampering_rejected':tamper,'key_substitution_rejected':bindings}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent;source=root/'models/SmolLM2-135M';cache=root/'.cache/huggingface';basehash=sha((source/'model.safetensors').read_bytes())
    first=None;trials=[]
    for repeat in range(1,6):
        data=setup(basehash);result=crypto_trial(data);trials.append({'repeat':repeat,**result})
        print('CRYPTO',repeat,json.dumps(result),flush=True)
        if repeat==1:first=data
    dump(out/'crypto_metrics.json',{'trials':trials,'scope':'five independent small-payload crypto trials; three real model deliveries separately'})
    gate,admission,(key,public,shares,proofs,keyctx),keys,recipients,public_registration=first
    t=Tardos(n=3);seeds=[replica_seed(admission,r) for r in recipients];words=[t.encode(s) for s in seeds]
    ids=[f'C-E0-U{i}' for i in (1,2,3)]
    public_registration.update(replica_ids=ids,seeds=seeds,words=[w.tolist() for w in words],p=t.p.tolist(),threshold=t.threshold)
    dump(out/'registration_before_training.json',public_registration)
    lab=ModelLab(str(source),cache,'cpu',2,True);data=corpus(cache,True,256);prompts=natural_triggers(lab,data['triggers'],3)
    dump(out/'prompts.json',prompts)
    print('PREPARING baseline and shared frozen caches',flush=True)
    baseline,predictions=utility(lab,data['test']);features=lab.features(prompts);cal=training_cache(lab,data['calibration'])
    metrics={'status':'running','base_sha256':basehash,'baseline':baseline,'data_hashes':{k:[sha(x.encode()) for x in v] for k,v in data.items()},
             'model_training_count':3,'steps':1200,'rows':[],'crypto_scope':'5 independent fixtures; real full model envelopes measured once each'}
    def save():dump(out/'model_metrics.json',metrics)
    def measure_box(active,name,strict=False,operator=None):
        selected=prompts[:1] if strict else prompts;protocol='single' if strict else 'batch'
        server=make_server(lambda ps:margins(active,ps),port=0,max_prompts=1 if strict else 768,response_field='score_differences')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();body=canonical({'protocol':protocol,'prompts':selected})
        url=f'http://127.0.0.1:{server.server_address[1]}/query';client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def send():return client.open(urllib.request.Request(url,body,{'Content-Type':'application/json'}),timeout=600)
        try:
            with send() as response:status=response.status;answer=json.load(response)
            try:
                with send() as response:second=response.status
            except urllib.error.HTTPError as error:second=error.code
        finally:server.shutdown();server.server_close();thread.join()
        values=answer['score_differences']
        # The black-box client consumes only HTTP responses and the prior codebook.
        # A strict single input yields insufficient evidence, never pad/guess 256 bits.
        if len(values)==768:
            bits=aggregate(values,3);decoded=t.decode(words,bits);scores=t.scores(words,bits).tolist()
        else:bits=np.full(256,-1);decoded=[];scores=None
        result={'endpoint':name,'simulated_operator':operator,'protocol':protocol,'inputs':len(selected),'first_http':status,'second_http':second,
                'returned_scores':len(values),'decoded_replicas':[ids[i] for i in decoded],'registered_custodians':[i+1 for i in decoded],
                'observed_bits':int((bits>=0).sum()),'correct_bits_per_replica':[int((bits==w).sum()) for w in words],
                'scores':scores,'classification':'登记副本来源线索' if decoded else '证据不足或无命中','individual_leaker_proven':False}
        dump(out/(name+'_http.json'),{'request_sha256':sha(body),'response':answer,'result':result})
        return result
    metrics['negative_control']=measure_box(lab,'clean_negative');save()
    for i,(replica,record) in enumerate(zip(ids,recipients)):
        gate.require_admitted(admission);print('TRAIN',replica,flush=True);torch.manual_seed(93001+i)
        training=embed_natural(lab,features,np.repeat(words[i],3),cal);adapter=out/(replica+'.pt');torch.save(lab.snapshot(),adapter)
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_STORED) as z:
            for path in sorted(source.iterdir()):
                if path.is_file():z.write(path,'base/'+path.name)
            z.write(adapter,'adapter.pt');z.writestr('replica.json',json.dumps({'replica_id':replica,'seed':seeds[i],'recipient_binding':record}))
        payload=buffer.getvalue();buffer.close();del buffer
        package=seal_copy(gate,admission,record,key,payload);expected=context_for(admission,record,sha(payload));del payload
        (out/(replica+'.aesgcm')).write_bytes(package['ciphertext'])
        dump(out/(replica+'_envelope.json'),{'context':package['context'],'wrapped_key':package['wrapped_key'].hex(),'cipher_sha256':sha(package['ciphertext'])})
        recovered=recover(public,shares,proofs,keyctx)
        cross={str(j+1):rejects(lambda j=j:open_copy(recovered,keys[j],package,expected)) for j in range(3) if j!=i}
        no_session=rejects(lambda:open_copy(key+1,keys[i],package,expected))
        plaintext=open_copy(recovered,keys[i],package,expected);same_hash=sha(plaintext)==expected['payload_sha256']
        with tempfile.TemporaryDirectory(dir=out) as temp:
            temp=Path(temp)
            with zipfile.ZipFile(io.BytesIO(plaintext)) as z:z.extractall(temp)
            del plaintext
            active=ModelLab(str(temp/'base'),cache,'cpu',2,True);active.restore(torch.load(temp/'adapter.pt',weights_only=True,map_location='cpu'))
            batch=measure_box(active,replica+'_batch');single=measure_box(active,replica+'_strict',strict=True)
            u,_=utility(active,data['test'],predictions)
            row={'replica':replica,'recipient':i+1,'training':training,'cipher_sha256':sha(package['ciphertext']),
                 'same_session_key':recovered==key,'owner_decrypt_hash_matches':same_hash,'cross_key_rejected':cross,'no_session_key_rejected':no_session,
                 'batch':batch,'strict_single':single,'ppl':u['perplexity'],'ppl_increase_pct':100*(u['perplexity']/baseline['perplexity']-1)}
            if i==0:
                # Same U1 plaintext served under a simulated U2 operator. This is
                # a controlled transfer, not proof of an actual person's behavior.
                metrics['transfer_control']=measure_box(active,'U1_copy_served_by_U2',operator=2)
            metrics['rows'].append(row);save();print('RESULT',json.dumps(row),flush=True)
            del active
        del package;gc.collect()
    metrics['status']='completed';save();print('COMPLETE',flush=True)

if __name__=='__main__':main()
