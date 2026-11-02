"""Remeasure a strict one-input box using saved adapters; preserve mistaken extra batch trials."""
import argparse,json,threading,urllib.request,urllib.error,gc
from pathlib import Path
import torch
from tvmc.crypto import canonical
from tvmc.personal_delivery import sha
from tvmc.model import ModelLab
from tvmc.robustness import margins
from serve import make_server


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();out=a.run_dir
    metrics=json.loads((out/'model_metrics.json').read_text())
    if metrics['status']!='completed':raise ValueError('finish training first')
    root=Path(__file__).resolve().parent;prompt=json.loads((out/'prompts.json').read_text())[0]
    rows=[]
    for row in metrics['rows']:
        replica=row['replica'];lab=ModelLab(str(root/'models/SmolLM2-135M'),root/'.cache/huggingface','cpu',2,True)
        lab.restore(torch.load(out/(replica+'.pt'),map_location='cpu',weights_only=True))
        server=make_server(lambda ps:margins(lab,ps),port=0,max_prompts=1,response_field='score_differences')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        body=canonical({'protocol':'single','prompts':[prompt]});url=f'http://127.0.0.1:{server.server_address[1]}/query'
        client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def send():return client.open(urllib.request.Request(url,body,{'Content-Type':'application/json'}),timeout=600)
        try:
            with send() as r:first=r.status;answer=json.load(r)
            try:
                with send() as r:second=r.status
            except urllib.error.HTTPError as e:second=e.code
        finally:server.shutdown();server.server_close();thread.join()
        assert len(answer['score_differences'])==1 and first==200 and second==409
        measured={'endpoint':replica+'_strict','protocol':'single','inputs':1,'first_http':first,'second_http':second,
                  'returned_scores':1,'decoded_replicas':[],'registered_custodians':[],'observed_bits':0,
                  'classification':'证据不足：一个分数不能满足256位读出协议','individual_leaker_proven':False}
        previous=out/(replica+'_strict_http.json')
        if previous.exists():
            old=json.loads(previous.read_text())
            if old['result']['protocol']=='batch':
                previous.rename(out/(replica+'_extra_batch_http.json'))
                row['extra_batch_due_to_initial_test_configuration']=row['strict_single']
        previous.write_text(json.dumps({'request_sha256':sha(body),'response':answer,'result':measured},ensure_ascii=False,indent=2)+'\n')
        row['strict_single']=measured;rows.append(measured)
        print(json.dumps(measured,ensure_ascii=False),flush=True);del lab;gc.collect()
    metrics['strict_single_correction']='Initial test accidentally used batch; preserved as extra_batch. Strict tests were rerun with saved unchanged adapters and exactly one prefix.'
    (out/'model_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    (out/'strict_single_retest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
