"""Replay archived real API evidence against an updated legitimate registry.
Only the endpoint exclusion is bypassed in the baseline; crypto checks stay on.
"""
import json,sys,copy
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tvmc.pki import Trust,Registry,issuer_key
from tvmc.paper_workflow import Ledger
from tvmc.paper_trace import judge_file,judge_endpoint
from tvmc.paper_threshold import ThresholdPublic
from tvmc.evidence import verify_quorum
from tvmc.crypto import canonical
source=Path(sys.argv[1]);root=Path(sys.argv[2]);out=Path(sys.argv[3])
data=json.loads((source/'trust.json').read_text());trace=json.loads((source/'trace.json').read_text());api=json.loads((source/'api_evidence.json').read_text())
reg=Registry(Trust(root.read_text(),data['pki']['crl'],data['pki']['issuer_certificate']));reg.update({int(k):v for k,v in data['registry'].items()})
ledger=Ledger();ledger.entries=[json.loads(l) for l in (source/'public_journal.jsonl').read_text().splitlines()]
assert ledger.check(data['checkpoint']['head']['chain'])
c=SimpleNamespace(ledger=ledger,registry=reg,issuer_pk=issuer_key(reg),code_key=ThresholdPublic(**data['code_key']),release_key=ThresholdPublic(**data['release_key']))
normal=judge_endpoint(trace,api['target'],api['probes'],api['records'],c,ledger.entries[-1]['hash'])
ledger.append('legitimate_endpoint',dict(member=1,endpoint=api['target']['endpoint']))
reader=lambda _:verify_quorum(api['records'],reg,3,api['target'],api['probes'],trace['group'])
off=judge_file(trace,canonical(api['target']),reader,ledger.entries,ledger.entries[-1]['hash'],c.issuer_pk,reg,c.code_key)
on=judge_endpoint(trace,api['target'],api['probes'],api['records'],c,ledger.entries[-1]['hash'])
result=dict(normal_unregistered_endpoint_accepted=normal,registered_endpoint_without_exclusion_accepted=off,registered_endpoint_with_exclusion_accepted=on,scope='archived real API evidence rechecked after endpoint registration; no new live investigation')
out.write_text(json.dumps(result,indent=2));print(result)
assert normal and off and not on
