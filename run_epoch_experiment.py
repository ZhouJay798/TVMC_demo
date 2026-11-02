"""Six independently delivered epoch models, actual C2 HTTP reads, five crypto repeats."""
import argparse, hashlib, io, json, pickle, threading, urllib.request, urllib.error, zipfile, time, os
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from tvmc.crypto import scalar, digest, canonical, prove, verify_proof
from tvmc.epochs import create
from tvmc.bridge import identity_seed
from tvmc.tardos import Tardos
from tvmc.model import ModelLab
from tvmc.robustness import corpus,natural_triggers,training_cache,embed_natural,margins,aggregate,utility
from serve import make_server
from tvmc.cross_session import session_leads


def write(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent;source=root/'models/SmolLM2-135M';cache=root/'.cache/huggingface'
    t=Tardos(n=6);records=[];sessions={};words=[];identity_history=[]
    for epoch in range(3):
        identities=[scalar() for _ in range(4)];identity_history.append(identities)
        for series,ids in [('A',(1,2)),('B',(2,3,4))]:
            sid=f'{series}-E{epoch}';key,session=create(sid,epoch,ids,identities);sessions[sid]=(key,session)
            seed=identity_seed(sid,epoch,ids,session.public);word=t.encode(seed);words.append(word)
            context={'domain':'TVMC/delivery-registration/v1','sid':sid,'epoch':epoch,'participants':ids,
                     'base_sha256':sha(source/'model.safetensors'),'seed':seed}
            proofs=[prove(session.public,s.share,context) for s in session.shares]
            assert all(verify_proof(session.public,x,context) for x in proofs)
            records.append({'sid':sid,'epoch':epoch,'participants':ids,'seed':seed,'public':asdict(session.public),
                            'context':context,'proofs':proofs,'word':word.tolist()})
    write(out/'registration_before_training.json',{'records':records,'threshold':t.threshold,'p':t.p.tolist(),
        'encoding':'SHA256 canonical JSON [domain,sid,epoch,sorted [custodian id,identity commitment]]'})
    private=out/'private_setup.pkl'
    with private.open('wb') as f: pickle.dump((sessions,identity_history),f)
    private.chmod(0o600)
    lab=ModelLab(str(source),cache,'cpu',4,True);data=corpus(cache,True,256)
    prompts=natural_triggers(lab,data['triggers'],3);assert len(prompts)==768
    write(out/'prompts.json',prompts)
    metrics={'status':'running','model_revision':'93efa2f097d58c2a74874c7e644dbc9b0cee75a2',
        'base_sha256':sha(source/'model.safetensors'),'training_repeats':1,'steps':1200,'rows':[],
        'data_hashes':{k:[hashlib.sha256(x.encode()).hexdigest() for x in v] for k,v in data.items()},
        'threshold':t.threshold,'torch':torch.__version__,'device':lab.device}
    def save():write(out/'model_metrics.json',metrics)
    print('Preparing fixed data, utility baseline and frozen training caches',flush=True)
    base,base_predictions=utility(lab,data['test']);metrics['baseline']=base
    features=lab.features(prompts);cal=training_cache(lab,data['calibration']);save()
    def measure(sid,expected=None):
        server=make_server(lambda ps:margins(lab,ps),port=0,max_prompts=768,response_field='score_differences')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        body=canonical({'protocol':'batch','prompts':prompts})
        url=f'http://127.0.0.1:{server.server_address[1]}/query'
        # Explicit localhost client, independent of any external HTTP proxy.
        client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def query():return client.open(urllib.request.Request(url,body,{'Content-Type':'application/json'}),timeout=600)
        try:
            with query() as response: first=response.status;answer=json.load(response)
            try:
                with query() as response: second=response.status
            except urllib.error.HTTPError as e:second=e.code
        finally: server.shutdown();server.server_close();thread.join()
        values=answer['score_differences'];assert len(values)==768
        observed=aggregate(values,3);decoded=t.decode(words,observed)
        u,_=utility(lab,data['test'],base_predictions)
        row={'model':sid,'expected_session':None if expected is None else records[expected]['sid'],
            'decoded_sessions':[records[i]['sid'] for i in decoded],
            'session_leads':session_leads([records[i]['sid'] for i in decoded],records),
            'decoded_custodians':[records[i]['participants'] for i in decoded],
            'scores':t.scores(words,observed).tolist(),'observed_bits':int((observed>=0).sum()),
            'correct_bits':None if expected is None else int((observed==words[expected]).sum()),
            'correct_localization':decoded==([] if expected is None else [expected]),
            'wrong_session_hits':[records[i]['sid'] for i in decoded if i!=expected],
            'first_http':first,'second_http':second,'utility':u,'ppl_increase':u['perplexity']/base['perplexity']-1}
        write(out/f'{sid}_http.json',{'request_sha256':hashlib.sha256(body).hexdigest(),'input_count':768,
            'response':answer,'second_http':second,'observed':observed.tolist()})
        metrics['rows'].append(row);save();print(json.dumps(row),flush=True)
    measure('clean_negative')
    for index,rec in enumerate(records):
        sid=rec['sid'];torch.manual_seed(71001+index)
        print('TRAIN',sid,'1200 steps',flush=True);started=time.monotonic()
        training=embed_natural(lab,features,np.repeat(words[index],3),cal)
        state=lab.snapshot();adapter=out/f'{sid}.pt';torch.save(state,adapter)
        # Complete reloadable delivery: original immutable backbone, tokenizer and trained adapter.
        archive=out/f'{sid}.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for path in sorted(source.iterdir()):
                if path.is_file():z.write(path,'base/'+path.name)
            z.write(adapter,'adapter.pt');z.writestr('registration.json',json.dumps(rec));
            z.writestr('format.json',json.dumps({'format':'TVMC base plus HeadLoRA v1','base_sha256':metrics['base_sha256']}))
        lab.head.reset();lab.restore(torch.load(adapter,map_location='cpu',weights_only=True))
        measure(sid,index)
        metrics['rows'][-1].update(training=training,training_seed=71001+index,
            train_delivery_measure_seconds=time.monotonic()-started,bundle_sha256=sha(archive),adapter_sha256=sha(adapter))
        save()
    metrics['status']='completed';save()

if __name__=='__main__':main()
