#!/usr/bin/env python3
"""TVMC v2: actual model shared-carrier issuance, t/n release and tracing."""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
os.environ.setdefault("HF_HOME",os.path.abspath(".cache/huggingface"))
from pathlib import Path
import secrets
import shutil
import tempfile
import time
import zipfile
import numpy as np
import torch
from tvmc.shared_carriers import CarrierLab,split_corpus
from tvmc.paper_workflow import Custody,sha
from tvmc.paper_trace import FileTracer,judge_file,escrow_parameters,disclose_parameters
from tvmc.stream_delivery import issue_file,receive_file,file_hash
from tvmc.transparency import Journal,verify_checkpoint,verify_extension
from tvmc.crypto import digest
from tvmc.tardos import Tardos


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def archive_model(directory,target):
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(Path(directory).iterdir()):
            if p.is_file():z.write(p,p.name)


def unpack_model(artifact,directory):
    with zipfile.ZipFile(artifact) as z:
        for name in z.namelist():
            if Path(name).name!=name or name.startswith('.'):raise ValueError('unsafe model archive')
        z.extractall(directory)


def model_reader(probes,bits,repeats,threads,root):
    def read(artifact):
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            unpack_model(artifact,tmp);lab=CarrierLab(tmp,threads=threads);lab.repeats=repeats
            word,_=lab.read(probes)
            del lab;gc.collect()
            return word[:bits].tolist()
    return read


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,default=Path('models/SmolLM2-135M-Instruct-Q4_1-converted'))
    ap.add_argument('--corpus',type=Path,default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--bits',type=int,default=128)
    ap.add_argument('--screen-bits',type=int,default=16)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--owner-steps',type=int,default=12)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--attacks',action='store_true')
    ap.add_argument('--sweep',action='store_true',help='Run optional version truncation diagnostics (also enabled by --attacks)')
    ap.add_argument('--evidence',choices=['file','api','relay'],default='file')
    args=ap.parse_args();out=args.output.resolve()
    if out.exists():raise SystemExit('Choose a new output directory; previous evidence must not be overwritten')
    if not args.model.is_dir() or not args.corpus.is_file():raise SystemExit('Local model/corpus missing; see README preparation instructions')
    if args.bits<64 or args.screen_bits<16:raise SystemExit('At least 64 attribution bits and 16 screening bits required')
    out.mkdir(parents=True,mode=0o700);start=time.time();checks={};events=[]
    def event(text):print(text,flush=True);events.append(dict(seconds=time.time()-start,event=text));dump(out/'progress.json',events)
    os.environ.setdefault('HF_HOME',str(out/'hf-cache'))
    lab=CarrierLab(args.model,threads=args.threads)
    train,generic,probes,utility=split_corpus(args.corpus,(args.bits+args.screen_bits)*args.repeats)
    # Calibration split is separate from final utility evaluation.
    corpus_text=args.corpus.read_text();N=len(corpus_text);part=corpus_text[int(.75*N):int(.8*N)]
    calibration_texts=[part[i:i+200] for i in range(0,min(len(part)-200,3200),200)]
    base_quality,_=lab.utility(utility)
    owner_training=lab.owner_train(train,steps=args.owner_steps)
    owner_quality,owner_preds=lab.utility(utility)
    _,calibration_preds=lab.utility(calibration_texts)
    event('Owner training complete; building shared carriers')
    spec=lab.build(probes,generic,bits=args.bits,screen_bits=args.screen_bits,repeats=args.repeats,margin=3.,ridge=.1)
    # Carrier acceptance uses unbiased test words, independent of issued Tardos words.
    test_rng=np.random.default_rng(991);acceptance=[];gain=None;calibration=[]
    testwords=test_rng.integers(0,2,(3,args.bits+args.screen_bits))
    for candidate in (.6,.8,1.):
        rows=[]
        for i,w in enumerate(testwords):
            lab.apply(w,candidate);obs,_=lab.read();q,_=lab.utility(calibration_texts,calibration_preds)
            rows.append(dict(test=i,error_fraction=float(np.mean(obs!=w)),utility=q))
        calibration.append(dict(gain=candidate,tests=rows))
        if all(r['error_fraction']<=.02 and r['utility']['top1_agreement']>=.85 for r in rows):
            gain=candidate;acceptance=rows;break
    dump(out/'carrier_calibration.json',calibration)
    dump(out/'carrier_acceptance.json',acceptance)
    if gain is None:
        dump(out/'report.json',dict(status='carrier_acceptance_failed',calibration=calibration))
        raise SystemExit('Carrier/readout/utility acceptance failed; no copies issued')
    event('Independent carrier acceptance passed')
    from tvmc.version_attacks import sweep
    version_sweep=sweep(lab,testwords[0],utility,owner_preds,gain) if args.sweep or args.attacks else dict(status='not_run',passes=None,scope='normal-flow demo; run --sweep to assess truncation')
    dump(out/'version_sweep.json',version_sweep)
    event(f'Version truncation sweep recorded; validated={version_sweep["passes"]}')
    # Fresh production word randomness; seed is kept only in private experiment state.
    seed=secrets.randbits(64);rng=np.random.default_rng(seed)
    tardos=Tardos(length=args.bits,n=3,seed=seed)
    words=(rng.random((3,args.bits))<tardos.p).astype(np.int8)
    screen=rng.integers(0,2,args.screen_bits).astype(np.int8)
    allwords=np.concatenate([words,np.tile(screen,(3,1))],axis=1)
    journal=Journal(out/'public_journal.jsonl');c=Custody(ledger=journal)
    journal.append('version_acceptance',dict(model=str(args.model.name),bits=args.bits,repeats=args.repeats,tests=acceptance,
        truncation_sweep=version_sweep,limitations=['SmolLM2 instead of GPT-2','finite test codewords, not exhaustive acceptance','truncation validation below']))
    reader_spec=dict(kind='causal-lm-logit-difference' if args.evidence=='file' else 'signed-api-logit-difference',architecture='SmolLM2',bits=args.bits,repeats=args.repeats,delta=1.,probe_hash=digest(spec[:args.bits*args.repeats]))
    tracer=FileTracer(c,tardos.p,tardos.threshold,reader_spec=reader_spec)
    attr_probes=spec[:args.bits*args.repeats];screen_probes=spec[args.bits*args.repeats:]
    attr_cap=escrow_parameters(c,tracer.group,dict(probes=attr_probes,repeats=args.repeats,delta=1.))
    screen_group='screen-1';screen_cap=escrow_parameters(c,screen_group,dict(probes=screen_probes,word=screen.tolist(),repeats=args.repeats,minimum_matches=math.ceil(.75*args.screen_bits)))
    delivered=[];copy_rows=[]
    for i,w in enumerate(allwords,1):
        cid=f'v2-U{i}';tracer.enroll(cid,words[i-1]);lab.apply(w,gain)
        obs,_=lab.read();quality,_=lab.utility(utility,owner_preds)
        rawdir=out/'issuer'/cid;rawdir.parent.mkdir(exist_ok=True,mode=0o700);lab.save_copy(rawdir)
        archive=rawdir.with_suffix('.zip');archive_model(rawdir,archive)
        package=issue_file(c,i,archive,out/'transit'/cid,cid)
        request=c.request(c.members[i-1],package);approvers=[m for m in c.members if m.i!=i][:3]
        for m in approvers:c.intent(m,request,package)
        approvals=[c.approval(m,request,package) for m in approvers]
        inner=c.release(request,package,approvals)
        dest=receive_file(c,c.members[i-1],package,inner,out/'recipients'/cid/'model.zip')
        checks[f'U{i}_delivery_exact']=file_hash(archive)==file_hash(dest)
        checks[f'U{i}_useful']=quality['top1_agreement']>=.85
        checks[f'U{i}_fingerprint_readable']=float(np.mean(obs[:args.bits]!=words[i-1]))<=.10
        row=dict(recipient=i,cid=cid,approvers=[m.i for m in approvers],error_fraction=float(np.mean(obs[:args.bits]!=words[i-1])),scores=tardos.scores(words,obs[:args.bits]).tolist(),quality=quality)
        copy_rows.append(row);delivered.append(dest)
        # Cleanup issuance plaintext artifacts; Python cannot promise secure memory erasure.
        archive.unlink();shutil.rmtree(rawdir)
        dump(out/'copies.json',copy_rows);event(f'U{i} actual model issued, independently threshold-released, received; bit errors={row["error_fraction"]:.3f}')
    attacks=[]
    if args.attacks:
        from tvmc.version_attacks import evaluate_attacks
        attacks=evaluate_attacks(lab,allwords,words,tardos,utility,owner_preds,args.bits,gain,out/'attacks.json',event,train)
    # Scope limitation is committed, never turn a failing attack into a success.
    journal.append('version_transform_results',dict(results=attacks,traceability_scope='only measured passing transforms; public-base truncation may fail'))
    api_details=None
    if args.evidence=='file':
        suspect=delivered[0];journal.append('evidence_commitment',dict(group=screen_group,file_sha256=sha(suspect)))
        screen_open=disclose_parameters(c,screen_group,screen_cap,[2,3,4],sha(suspect))
        screening=model_reader(screen_open['probes'],args.screen_bits,args.repeats,args.threads,out)(suspect)
        matched=sum(a==b for a,b in zip(screening,screen_open['word']))
        checks['screening_passed']=matched>=screen_open['minimum_matches']
        if not checks['screening_passed']:raise SystemExit('Suspect failed screening; attribution group not disclosed')
        tracer.commit_file(suspect)
        opened=disclose_parameters(c,tracer.group,attr_cap,[2,3,4],sha(suspect))
        read=model_reader(opened['probes'],args.bits,args.repeats,args.threads,out)
        event('Screening passed; committed suspect file; quorum disclosed attribution probes')
        result=tracer.trace_file(suspect,read,[2,3,4]);checks['source_U1_only']=result['recipients']==[1]
        checkpoint=journal.checkpoint(c.issuer,c.members[1:4])
        checks['public_judge_verified']=judge_file(result,suspect,read,journal.entries,checkpoint['head']['chain'],c.issuer_pk,c.registry,c.code_key)
    else:
        from tvmc.api_server import Endpoint
        from tvmc.evidence import collect_quorum,verify_quorum,https_query,ServedLog,verify_rebuttal
        from tvmc.paper_trace import judge_endpoint,rescore_rebuttal
        from tvmc.crypto import canonical
        tmp=out/'api_received_model';tmp.mkdir();unpack_model(delivered[0],tmp)
        api_lab=CarrierLab(tmp,threads=args.threads);served=ServedLog(c.members[0],journal)
        legitimate=Endpoint(out/'tls_source',api_lab,served)
        suspect_server=Endpoint(out/'tls_suspect',forward=legitimate.target) if args.evidence=='relay' else legitimate
        target=suspect_server.target;descriptor=canonical(target)
        if args.evidence=='relay':journal.append('legitimate_endpoint',dict(member=1,endpoint=legitimate.target['endpoint']))
        try:
            journal.append('endpoint_commitment',dict(group=screen_group,target=target))
            journal.append('evidence_commitment',dict(group=screen_group,file_sha256=sha(descriptor),kind='endpoint_descriptor'))
            screen_open=disclose_parameters(c,screen_group,screen_cap,[2,3,4],sha(descriptor))
            screening,_=collect_quorum(c,target,screen_open['probes'],https_query,[2,3,4],screen_group,args.repeats)
            checks['screening_passed']=sum(a==b for a,b in zip(screening,screen_open['word']))>=screen_open['minimum_matches']
            if not checks['screening_passed']:raise ValueError('API failed screening')
            journal.append('endpoint_commitment',dict(group=tracer.group,target=target))
            tracer.commit_file(descriptor)
            opened=disclose_parameters(c,tracer.group,attr_cap,[2,3,4],sha(descriptor))
            observed,records=collect_quorum(c,target,opened['probes'],https_query,[2,3,4],tracer.group,args.repeats)
            read=lambda _:verify_quorum(records,c.registry,3,target,opened['probes'],tracer.group,args.repeats)
            result=tracer.trace_file(descriptor,read,[2,3,4])
            checks['source_U1_only']=result['recipients']==[1]
            checks['public_judge_verified']=judge_endpoint(result,target,opened['probes'],records,c,journal.entries[-1]['hash'])
            api_details=dict(target=target,collectors=[2,3,4],words_identical=True)
            if args.evidence=='relay':
                rebut=served.rebut(opened['probes'],(target['start'],target['end']))
                anchors=[e for e in journal.entries if e['kind']=='served-head']
                trace_seq=next(e['seq'] for e in journal.entries if e['kind']=='trace_result')
                anchor=anchors[-1]
                if anchor['seq']>=trace_seq:raise ValueError('late rebuttal anchor')
                erased=verify_rebuttal(rebut,anchor['body'],c.registry,opened['probes'],(target['start'],target['end']),delta=1800.)
                body=dict(group=tracer.group,cid='v2-U1',erased=erased,digest=digest(rebut))
                journal.append('verified_rebuttal',body)
                rescore=rescore_rebuttal(c,tracer,result,'v2-U1',erased,[2,3,4],digest(rebut))
                checks['relay_not_mislabeled_as_weight_leak']=not rescore['retained']
                api_details['rebuttal']=rescore
                dump(out/'rebuttal.json',dict(rebuttal=rebut,anchor=anchor['body'],result=rescore))
            checkpoint=journal.checkpoint(c.issuer,c.members[1:4])
            dump(out/'api_evidence.json',dict(target=target,records=records,probes=opened['probes']))
        finally:
            if suspect_server is not legitimate:suspect_server.close()
            legitimate.close();del api_lab;gc.collect()
    checks['witness_checkpoint']=verify_checkpoint(checkpoint,c.issuer_pk,c.registry,3)
    trust=dict(issuer_pk=c.issuer_pk.hex(),registry=c.registry,code_key=c.code_key.__dict__,release_key=c.release_key.__dict__,checkpoint=checkpoint)
    dump(out/'trust.json',trust);dump(out/'trace.json',result)
    report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,seconds=time.time()-start,
        model=str(args.model.name),model_parameters=sum(p.numel() for p in lab.model.parameters()),corpus_sha256=file_hash(args.corpus),
        evidence_kind=args.evidence,api_details=api_details,bits=args.bits,screen_bits=args.screen_bits,repeats=args.repeats,members=5,threshold=3,
        version_sweep=version_sweep,carrier_gain=gain,owner_training=owner_training,base_quality=base_quality,owner_quality=owner_quality,acceptance=acceptance,copies=copy_rows,
        attack_results=attacks,source_candidates=result['recipients'],final_attribution=[] if api_details and api_details.get('rebuttal',{}).get('retained') is False else result['recipients'],attribution=result['recipients'],scores=result['scores'],score_threshold=tardos.threshold,
        limitations=['Research TDH2/Pedersen-Feldman port over 2048-bit group, not ristretto255 or independently audited',
          'Local role simulation; no hostile-OS isolation, independent witnesses or L3 enclave',
          'Measured small-scale SmolLM2 experiment, not paper 100-copy/2048-bit configuration',
          'One attribution group; robustness probes are issuer diagnostics, not new adjudications',
          'API TLS pins and witnesses belong to local simulated roles, not external attestation',
          'Secure erasure and deployment restart key storage not provided'])
    dump(out/'report.json',report);event('Public judge verified file readout / signed API evidence and encrypted scores')
    print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if all(checks.values()) else 2

if __name__=='__main__':raise SystemExit(main())
