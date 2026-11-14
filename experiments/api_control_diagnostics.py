"""Real HTTPS controls using previously issued copies and disclosed probes.
Diagnostic reuse only: NOT a new single-use confidential L2 investigation.
"""
import sys,json,time,gc,copy,csv,ssl,http.client,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parents[1]/'.cache/huggingface'))
import numpy as np
from tvmc.paper_workflow import Custody,sign
from tvmc.shared_carriers import CarrierLab
from tvmc.api_server import Endpoint
from tvmc.evidence import collect_quorum,verify_quorum,read_scores,https_query,ServedLog,verify_rebuttal
from tvmc.tardos import Tardos
from run_experiment import unpack_model


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    entries=[json.loads(l) for l in (source/'public_journal.jsonl').read_text().splitlines()]
    attr=next(e['body']['opening']['parameters'] for e in entries if e['kind']=='probe_disclosure' and e['body']['group']=='group-1')
    screen=next(e['body']['opening']['parameters'] for e in entries if e['kind']=='probe_disclosure' and e['body']['group']=='screen-1')
    params=next(e['body']['parameters'] for e in entries if e['kind']=='group_disclosed_and_retired')
    probes=attr['probes']+screen['probes'];bits=len(params['biases']);repeats=attr['repeats']
    paths={};words=[]
    for i in (1,2,3):
        directory=output/('reference-U'+str(i));unpack_model(source/f'recipients/v2-U{i}/model.zip',directory)
        paths['U'+str(i)]=directory
        lab=CarrierLab(directory,threads=2);lab.repeats=repeats;word,_=lab.read(probes)
        words.append(word[:bits]);del lab;gc.collect();print('reference ready',i,flush=True)
    words=np.asarray(words);assert np.isin(words,[0,1]).all()
    scorer=Tardos(bits,n=3);scorer.p=np.asarray(params['biases']);scorer.threshold=params['threshold']
    c=Custody();rows=[];controls=[]
    paths['unmarked']=Path('models/SmolLM2-135M-Instruct-Q4_1-converted')
    for name,path in paths.items():
        lab=CarrierLab(path,threads=2);served=ServedLog(c.members[0],c.ledger)
        server=Endpoint(output/('tls-'+name),lab,served)
        try:
            target=server.target;group='diagnostic-'+name
            c.ledger.append('endpoint_commitment',dict(group=group,target=target))
            observed,records=collect_quorum(c,target,probes,https_query,[2,3,4],group,repeats)
            scores=scorer.scores(words,observed[:bits]);matches=sum(a==b for a,b in zip(observed[bits:],screen['word']))
            hits=[i+1 for i,x in enumerate(scores) if x>=scorer.threshold] if matches>=screen['minimum_matches'] else []
            rows.append(dict(case=name,screen_matches=int(matches),screen_pass=bool(matches>=screen['minimum_matches']),scores=scores.tolist(),initial_hits=hits,final_hits=hits,collectors=3,kind='actual model HTTPS'))
            if name=='U1':
                # Same actual endpoint, fake certificate pin: insecure TLS accepts;
                # production pin-check refuses before sending the probe payload.
                from urllib.parse import urlsplit
                u=urlsplit(target['endpoint']);assert u.hostname=='127.0.0.1'
                conn=http.client.HTTPSConnection(u.hostname,u.port,context=ssl._create_unverified_context(),timeout=60)
                try:
                    conn.request('POST',u.path,json.dumps(dict(probes=probes[:3],collector=2)),{'Content-Type':'application/json'})
                    off_ok=conn.getresponse().status==200
                finally:conn.close()
                try:https_query(dict(target,tls_sha256='00'*32),probes[:3],2);on_ok=True
                except ValueError:on_ok=False
                controls.append(dict(case='TLS_identity_replacement',without_control_accepted=off_ok,with_control_accepted=on_ok))
                # Forge first collector's raw outputs to fingerprint U2.
                bad=copy.deepcopy(records)
                injected=[]
                for bit in list(words[1])+list(screen['word']):injected.extend([2. if bit else -2.]*repeats)
                bad[0]['body']['scores']=injected
                for signed in (False,True):
                    if signed:bad[0]['signature']=sign(c.members[1].identity,'api-evidence',bad[0]['body'])
                    naive=read_scores(bad[0]['body']['scores'],repeats)
                    naivehits=[i+1 for i in scorer.decode(words,naive[:bits])]
                    try:verify_quorum(bad,c.registry,3,target,probes,group,repeats);accepted=True
                    except ValueError:accepted=False
                    controls.append(dict(case='collector_disagreement' if signed else 'forged_API_responses',without_control_accepted=True,without_control_hits=naivehits,with_control_accepted=accepted))
                proxy=Endpoint(output/'tls-relay',forward=target)
                try:
                    proxy_target=proxy.target;g='diagnostic-relay';c.ledger.append('endpoint_commitment',dict(group=g,target=proxy_target))
                    obs,rs=collect_quorum(c,proxy_target,probes,https_query,[2,3,4],g,repeats)
                    initial=[i+1 for i in scorer.decode(words,obs[:bits])]
                    rebut=served.rebut(attr['probes'],(proxy_target['start'],proxy_target['end']))
                    anchor=next(e['body'] for e in reversed(c.ledger.entries) if e['kind']=='served-head')
                    erased=verify_rebuttal(rebut,anchor,c.registry,attr['probes'],(proxy_target['start'],proxy_target['end']),delta=1800.)
                    adjusted=list(obs[:bits])
                    for k in erased:adjusted[k]=-1
                    final=[i+1 for i in scorer.decode(words,adjusted)]
                    rows.append(dict(case='relay_U1',screen_matches=None,screen_pass=True,scores=scorer.scores(words,obs[:bits]).tolist(),initial_hits=initial,final_hits=final,erased_bits=len(erased),collectors=3,kind='actual HTTPS forwarder + anchored rebuttal'))
                    controls.append(dict(case='relay_without_rebuttal',without_control_accepted=bool(initial),with_control_accepted=bool(final),meaning='false weight-leak accusation'))
                finally:proxy.close()
            print(name,rows[-1],flush=True)
        finally:server.close();del lab;gc.collect()
        (output/'api_results.json').write_text(json.dumps(dict(rows=rows,controls=controls),ensure_ascii=False,indent=2))
    assert all(r['final_hits']==([int(r['case'][1:])] if r['case'] in ('U1','U2','U3') else []) for r in rows)
    assert all(c['without_control_accepted'] and not c['with_control_accepted'] for c in controls)
    print('ALL API DIAGNOSTIC EXPECTATIONS PASSED',flush=True)

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
