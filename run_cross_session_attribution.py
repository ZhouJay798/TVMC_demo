"""Replay local model observations and simulate mixed observations for session leads.

The registration file and locally stored HTTP observations are trusted inputs:
HTTP score transcripts are not signed collector receipts. Simulated code-bit
mixtures are NOT measurements on mixed models or an estimate of attack success.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from tvmc.crypto import verify_proof,canonical
from tvmc.epochs import EpochPublic
from tvmc.bridge import identity_seed
from tvmc.tardos import Tardos
from tvmc.cross_session import session_leads
from tvmc.robustness import aggregate


def validated_records(registration):
    records=registration['records']
    # Validates uniqueness, participant types, and epoch before any mapping.
    session_leads([],records)
    t=Tardos(n=len(records),length=len(registration['p']))
    t.p=np.array(registration['p']);t.threshold=registration['threshold']
    for r in records:
        pub=EpochPublic(**r['public']); ctx=r['context'];ids=r['participants'];proofs=r['proofs']
        if (set(pub.participants)!=set(ids) or
            ctx.get('sid')!=r['sid'] or ctx.get('epoch')!=r['epoch'] or
            ctx.get('participants')!=ids or ctx.get('seed')!=r['seed'] or
            identity_seed(r['sid'],r['epoch'],ids,pub)!=r['seed'] or
            len(proofs)!=len(ids) or {p['i'] for p in proofs}!=set(ids) or
            not all(verify_proof(pub,p,ctx) for p in proofs) or
            not np.array_equal(t.encode(r['seed']),r['word'])):
            raise ValueError('invalid registration for '+r['sid'])
    return records,t


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    registration=json.loads((args.run_dir/'registration_before_training.json').read_text())
    records,t=validated_records(registration);words=[r['word'] for r in records]
    def outcome(observed):
        indices=t.decode(words,observed);sids=[records[i]['sid'] for i in indices]
        return {'decoded_sessions':sids,'scores':t.scores(words,observed).tolist(),
                'leads':session_leads(sids,records)}
    observations={};actual=[]
    prompts=json.loads((args.run_dir/'prompts.json').read_text())
    request_hash=hashlib.sha256(canonical({'protocol':'batch','prompts':prompts})).hexdigest()
    for sid in [r['sid'] for r in records]+['clean_negative']:
        file=json.loads((args.run_dir/(sid+'_http.json')).read_text())
        obs=np.array(file['observed'],dtype=int)
        margins=np.array(file['response']['score_differences'],dtype=float)
        if (len(prompts)!=768 or len(margins)!=768 or not np.isfinite(margins).all() or
            file['input_count']!=768 or file['second_http']!=409 or file['request_sha256']!=request_hash or
            not np.array_equal(aggregate(margins,3),obs)):
            raise ValueError('invalid local HTTP transcript: '+sid)
        observations[sid]=obs
        actual.append({'model':sid,'source':'prior real model HTTP observation','result':outcome(obs)})
    a,b=observations['A-E0'],observations['B-E0'];rng=np.random.default_rng(48371);sim=[]
    for i in range(20):
        mixed=np.where(rng.integers(0,2,len(a)),a,b)
        sim.append({'scenario':f'interleave_{i+1:02d}','result':outcome(mixed)})
    for name,mixed in [('AND',a&b),('OR',a|b)]:sim.append({'scenario':name,'result':outcome(mixed)})
    result={'scope':'会话级线索；可信快照和本地HTTP观测作为输入，未验证HTTP采集方真实性。',
            'mix_scope':'先前真实模型读出位的模拟混合，不是实际混合模型或攻击成功率。',
            'threshold':t.threshold,'real_model_replays':actual,'simulated_bit_mixtures':sim}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for row in actual:print('REAL',row['model'],row['result']['decoded_sessions'],row['result']['leads']['rule'])
    for row in sim:print('SIM',row['scenario'],row['result']['decoded_sessions'],row['result']['leads']['rule'])

if __name__=='__main__':main()
