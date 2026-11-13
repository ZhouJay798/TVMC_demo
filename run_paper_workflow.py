#!/usr/bin/env python3
"""Reproducible additive new-paper prototype. Does not modify old experiments."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from tvmc.paper_workflow import Custody,sha
from tvmc.paper_trace import FileTracer,judge_file
from tvmc.tardos import Tardos


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('work/paper_workflow'))
    ap.add_argument('--asset',type=Path,help='Optional existing fingerprinted model archive: delivery-only check')
    ap.add_argument('--length',type=int,default=128)
    args=ap.parse_args()
    if args.output.exists():raise SystemExit('Output exists; choose a fresh directory to preserve history')
    if args.length<64:raise SystemExit('Use at least 64 bits for the scoring fixture')
    start=time.monotonic();args.output.mkdir(parents=True,mode=0o700)
    c=Custody(); print('dual key generation and five registrations complete',flush=True)
    tardos=Tardos(length=args.length,n=3)
    tracer=FileTracer(c,tardos.p,tardos.threshold)
    rng=np.random.default_rng(43)
    # Deterministic test-only words, NEVER deployment fingerprints.
    words=(rng.random((3,args.length))<tardos.p).astype(np.int8)
    # Shared orthogonal carriers in a tiny vector fixture: W_i = W + E0 + sum (2w-1)E_k.
    carriers=np.eye(args.length);base=np.zeros(args.length); delivered=[];checks={}
    for i,word in enumerate(words,1):
        cid=f'fixture-U{i}'
        tracer.enroll(cid,word)
        model=base+(2*word-1)@carriers
        payload=json.dumps({'kind':'synthetic-vector-not-LLM','weights':model.tolist()}).encode()
        package=c.issue(i,payload,cid)
        request=c.request(c.members[i-1],package)
        approvers=[m for m in c.members if m.i!=i][:3]
        for member in approvers:c.intent(member,request,package)
        approvals=[c.approval(member,request,package) for member in approvers]
        inner=c.release(request,package,approvals)
        recovered=c.receive(c.members[i-1],package,inner)
        checks[f'U{i}_delivery_exact']=recovered==payload
        delivered.append(recovered)
        print(f'U{i} released by {[m.i for m in approvers]}, received and verified',flush=True)
    tracer.commit_file(delivered[0])
    def read(blob):return (np.array(json.loads(blob)['weights'])>0).astype(int).tolist()
    result=tracer.trace_file(delivered[0],read,[2,3,4])
    expected=tardos.scores(words,np.array(read(delivered[0])))
    checks['encrypted_scores_match_plaintext']=all(abs(result['scores'][f'fixture-U{i+1}']-float(s)) <= (args.length+1)/2048 for i,s in enumerate(expected))
    checks['source_U1_only']=result['recipients']==[1]
    checks['ledger_chain_valid']=c.ledger.check()
    checks['public_judge_verified']=judge_file(result,delivered[0],read,c.ledger.entries,c.ledger.entries[-1]['hash'],c.issuer_pk,c.registry,c.code_key)
    if args.asset:
        payload=args.asset.read_bytes();package=c.issue(1,payload,'existing-model-delivery-only')
        request=c.request(c.members[0],package)
        for m in c.members[1:4]:c.intent(m,request,package)
        inner=c.release(request,package,[c.approval(m,request,package) for m in c.members[1:4]])
        recovered=c.receive(c.members[0],package,inner)
        checks['existing_model_bytes_unchanged']=sha(payload)==sha(recovered)
    report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
                elapsed_seconds=time.monotonic()-start,source='new-paper additive research prototype',
                model_scope='synthetic shared-carrier vector; optional old real model delivery only',
                scores=result['scores'],recipients=result['recipients'],n=5,t=3,
                limitations=['Feldman joint-keygen simulation, not paper Pedersen DKG',
                 'labeled DH-KEM prototype, NOT TDH2; no CCA claim',
                 'single-process actor simulation, no isolation or network',
                 'no GPT-2 shared-carrier editing or robustness validation',
                 'no probe escrow, API collector quorum, rebuttal, refresh or L3 enclave',
                 'local hash-chain ledger, no independent witness or external identity assurance'])
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (args.output/'public_transcript.json').write_text(json.dumps(c.ledger.entries,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if all(checks.values()) else 2

if __name__=='__main__':raise SystemExit(main())
