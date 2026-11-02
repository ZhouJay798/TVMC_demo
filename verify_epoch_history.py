"""Reload actual decrypted delivery packages and trace them using historical records."""
import argparse,hashlib,io,json,pickle,tempfile,threading,urllib.request,urllib.error,zipfile
from pathlib import Path
import numpy as np
import torch
from tvmc.crypto import decrypt,canonical,verify_proof
from tvmc.epochs import EpochPublic
from tvmc.bridge import identity_seed
from tvmc.model import ModelLab
from tvmc.robustness import margins,aggregate
from tvmc.tardos import Tardos
from serve import make_server
from tvmc.cross_session import session_leads


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();out=a.run_dir
    crypto=json.loads((out/'crypto_metrics.json').read_text())
    assert crypto['status']=='completed'
    # Public-only replay audits isolate the epoch field and separately target new deliveries.
    by_session={(r['repeat'],r['branch'],r['sid']):r for r in crypto['history']}
    replay=[]
    for item in crypto['history']:
        original=item['epochs'][0];public=EpochPublic(**original['public'])
        wrong_context=dict(original['context'],share_epoch=original['epoch']+1)
        replay.append({'repeat':item['repeat'],'branch':item['branch'],'sid':item['sid'],
            'test':'epoch_only_replay','rejected':not any(verify_proof(public,p,wrong_context) for p in original['proofs'])})
        for epoch in range(original['epoch']+1,3):
            target=item['sid'][0]+f'-E{epoch}'
            new=by_session[item['repeat'],item['branch'],target]['epochs'][0]
            replay.append({'repeat':item['repeat'],'branch':item['branch'],'sid':item['sid'],'target':target,
                'test':'new_delivery_replay','rejected':not any(verify_proof(EpochPublic(**new['public']),p,new['context']) for p in original['proofs'])})
    (out/'public_replay_audit.json').write_text(json.dumps(replay,indent=2))
    print('PUBLIC REPLAY AUDIT',len(replay),'rejected',sum(r['rejected'] for r in replay),flush=True)
    with (out/'private_setup.pkl').open('rb') as f:sessions,identities=pickle.load(f)
    registration=json.loads((out/'registration_before_training.json').read_text());records=registration['records']
    prompts=json.loads((out/'prompts.json').read_text());t=Tardos(n=6);words=[r['word'] for r in records];rows=[]
    for index,rec in enumerate(records):
        sid=rec['sid'];key,session=sessions[sid]
        print('RELOAD AND HISTORICAL HTTP',sid,flush=True)
        expected=hashlib.sha256((out/f'{sid}.zip').read_bytes()).hexdigest()
        aad={'sid':sid,'delivery_epoch':rec['epoch'],'payload_sha256':expected}
        plain=decrypt(key,(out/f'{sid}.aesgcm').read_bytes(),aad)
        actual=hashlib.sha256(plain).hexdigest()
        with tempfile.TemporaryDirectory(dir=out) as directory:
            path=Path(directory)
            with zipfile.ZipFile(io.BytesIO(plain)) as z:z.extractall(path)
            del plain
            lab=ModelLab(str(path/'base'),out/'cache','cpu',2,True)
            lab.restore(torch.load(path/'adapter.pt',weights_only=True,map_location='cpu'))
            server=make_server(lambda ps:margins(lab,ps),port=0,max_prompts=768,response_field='score_differences')
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_address[1]}/query'
            client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            body=canonical({'protocol':'batch','prompts':prompts})
            def query():return client.open(urllib.request.Request(url,body,{'Content-Type':'application/json'}),timeout=600)
            try:
                with query() as response:answer=json.load(response)
                try:
                    with query() as response:second=response.status
                except urllib.error.HTTPError as error:second=error.code
            finally:server.shutdown();server.server_close();thread.join()
            observed=aggregate(answer['score_differences'],3);decoded=t.decode(words,observed)
            prior=json.loads((out/f'{sid}_http.json').read_text())['observed']
            row={'sid':sid,'hash_matches':actual==expected,'loaded_from_decrypted_package':True,
                 'seed_matches_historical_registry':identity_seed(sid,rec['epoch'],rec['participants'],session.public)==rec['seed'],
                 'saved_registration_proofs_valid':all(verify_proof(session.public,p,rec['context']) for p in rec['proofs']),
                 'decoded_sessions':[records[i]['sid'] for i in decoded],
            'session_leads':session_leads([records[i]['sid'] for i in decoded],records), 'correct_localization':decoded==[index],
                 'correct_bits':int((observed==words[index]).sum()),'observed_bits':int((observed>=0).sum()),
                 'same_observed_bits':bool(np.array_equal(observed,prior)), 'second_http':second}
            rows.append(row)
            (out/f'{sid}_historical_http.json').write_text(json.dumps({'response':answer,'result':row},indent=2))
            (out/'historical_metrics.json').write_text(json.dumps({'status':'running','rows':rows},indent=2))
            print(json.dumps(row),flush=True);del lab
    # Logical deletion in an isolated local identity store, with historical public records retained.
    # The test harness deliberately retains captured secrets to test the compromise boundary.
    local_store={'E0':list(identities[0]),'E1':list(identities[1]),'E2':list(identities[2])}
    del local_store['E0'];del local_store['E1']
    (out/'historical_metrics.json').write_text(json.dumps({'status':'completed','rows':rows,
        'logical_erasure_simulated':list(local_store)==['E2'],'secure_erasure_claimed':False},indent=2))

if __name__=='__main__':main()
