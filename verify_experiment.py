"""Recompute an experiment verdict using public keys and evidence, no private keys."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from tvmc.paper_threshold import ThresholdPublic
from tvmc.paper_trace import judge_file,judge_endpoint
from tvmc.paper_workflow import sha
from tvmc.pki import Trust, Registry, issuer_key, member_key
from tvmc.transparency import Journal,verify_checkpoint
from run_experiment import model_reader


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path)
    ap.add_argument('--trust-sha256',help='Externally retained hash of trust.json; required unless --ca-root is supplied')
    ap.add_argument('--ca-root',type=Path,help='CA root provisioned through an independent trusted channel')
    ap.add_argument('--min-crl-number',type=int,default=0,help='Minimum CRL number retained outside the experiment directory')
    ap.add_argument('--threads',type=int,default=4);args=ap.parse_args();root=args.run
    trust_path=root/'trust.json'
    if args.trust_sha256 and sha(trust_path)!=args.trust_sha256:raise SystemExit('Trust anchor was replaced')
    trust=json.loads(trust_path.read_text());trace=json.loads((root/'trace.json').read_text())
    if 'pki' not in trust:raise SystemExit('Legacy artifact has no certificates; use its historical verifier')
    if args.ca_root is None and not args.trust_sha256:
        raise SystemExit('Supply an independently trusted --ca-root or --trust-sha256; refusing self-authenticated keys')
    root_pem=args.ca_root.read_text() if args.ca_root else trust['pki']['root_certificate']
    authority=Trust(root_pem,trust['pki']['crl'],trust['pki']['issuer_certificate'],args.min_crl_number)
    registry=Registry(authority);registry.update({int(k):v for k,v in trust['registry'].items()})
    for i in registry:
        member_key(registry,i);member_key(registry,i,'receiving')
    journal=Journal(root/'public_journal.jsonl')
    issuer=issuer_key(registry,bytes.fromhex(trust['issuer_pk']));code=ThresholdPublic(**trust['code_key']);release=ThresholdPublic(**trust['release_key'])
    checkpoint=trust['checkpoint']
    if not verify_checkpoint(checkpoint,issuer,registry,release.t):raise SystemExit('Checkpoint signatures invalid')
    if not journal.check(checkpoint['head']['chain']):raise SystemExit('Journal changed or rolled back')
    evidence_versions=[]
    if (root/'api_evidence.json').exists():
        api=json.loads((root/'api_evidence.json').read_text())
        evidence_versions=sorted({r['body'].get('evidence_version',1) for r in api['records']})
        public=SimpleNamespace(ledger=journal,registry=registry,issuer_pk=issuer,code_key=code,release_key=release)
        passed=judge_endpoint(trace,api['target'],api['probes'],api['records'],public,checkpoint['head']['chain'])
    else:
        disclosure=next(e['body'] for e in journal.entries if e['kind']=='probe_disclosure' and e['body']['group']==trace['group'])
        probes=disclosure['opening']['parameters']['probes'];repeats=disclosure['opening']['parameters']['repeats']
        artifact=root/'recipients/v2-U1/model.zip';reader=model_reader(probes,len(probes)//repeats,repeats,args.threads,root)
        passed=judge_file(trace,artifact,reader,journal.entries,checkpoint['head']['chain'],issuer,registry,code)
    print(json.dumps(dict(verified=passed,scope='independently pinned trust file' if args.trust_sha256 else 'externally provisioned CA root',
                         api_evidence_versions=evidence_versions or None,
                         api_completion_times_verified=(passed and evidence_versions==[2]) if evidence_versions else None,
                         recipients=trace['recipients'],rebuttal_present=(root/'rebuttal.json').exists()),ensure_ascii=False))
    if (root/'rebuttal.json').exists():
        # Separate API source finding from the post-rebuttal verdict.
        from tvmc.evidence import verify_rebuttal
        rebut=json.loads((root/'rebuttal.json').read_text());api=json.loads((root/'api_evidence.json').read_text())
        positions=verify_rebuttal(rebut['rebuttal'],rebut['anchor'],registry,api['probes'],(api['target']['start'],api['target']['end']),delta=1800.)
        from tvmc.paper_trace import judge_rebuttal
        trace_seq=next(e['seq'] for e in journal.entries if e['kind']=='trace_result' and e['body']['group']==trace['group'])
        anchors=[e for e in journal.entries if e['kind']=='served-head' and e['body']==rebut['anchor']]
        passed=passed and len(anchors)==1 and anchors[0]['seq']<trace_seq and judge_rebuttal(trace,rebut,positions,journal.entries,registry,code)
        print(json.dumps(dict(verified_relay_positions=len(positions),encrypted_rescore_verified=passed,retained=rebut['result']['retained']),ensure_ascii=False))
    return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
