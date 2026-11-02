"""Five independent cryptographic repetitions, both rotation branches, actual delivery bytes.
Only repeat 1 / rotate uses the six corresponding trained fingerprints. Other repetitions
re-encrypt those fixed model payloads to test cryptography, and are NOT new model trainings.
"""
import argparse,copy,hashlib,io,itertools,json,pickle,zipfile,time
from pathlib import Path
from dataclasses import asdict,replace
import torch
from cryptography.exceptions import InvalidTag
from tvmc.crypto import scalar,G,P,Q,encrypt,decrypt,digest,prove,verify_proof
from tvmc.epochs import create,contribution,refresh,rotate_session_identity,interpolate
from tvmc.bridge import identity_seed


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--wait-for-models',action='store_true');args=p.parse_args();out=args.run_dir
    with (out/'private_setup.pkl').open('rb') as f:first_sessions,first_identities=pickle.load(f)
    results=[];history=[];started=time.monotonic()
    def save():
        (out/'crypto_metrics.json').write_text(json.dumps({'status':'running','checks':results,'history':history},indent=2))
    def record(repeat,branch,sid,name,passed,detail=None):
        results.append({'repeat':repeat,'branch':branch,'sid':sid,'test':name,'passed':bool(passed),'detail':detail})
    def rejects(fn):
        try:fn();return False
        except (ValueError,InvalidTag):return True
    def sha(b):return hashlib.sha256(b).hexdigest()
    def matches(key,cipher,aad,expected):
        try:return sha(decrypt(key,cipher,aad))==expected
        except InvalidTag:return False
    registration=json.loads((out/'registration_before_training.json').read_text())['records']
    # Trusted pre-training snapshot is retained. Integrity is measured relative to this pinned digest.
    for repeat in range(1,6):
      for branch in ('rotate','keep'):
        print('CRYPTO',repeat,branch,flush=True)
        identities=[]
        for epoch in range(3):
            if repeat==1 and branch=='rotate':values=first_identities[epoch]
            elif epoch and branch=='keep':values=identities[0]
            else:values=[scalar() for _ in range(4)]
            identities.append(values)
        sessions={}
        for series,ids in [('A',(1,2)),('B',(2,3,4))]:
          for epoch in range(3):
            sid=f'{series}-E{epoch}'
            sessions[sid]=first_sessions[sid] if repeat==1 and branch=='rotate' else create(sid,epoch,ids,identities[epoch])
        record(repeat,branch,'all','independent_delivery_keys',len({k for k,s in sessions.values()})==6)
        seeds=[identity_seed(s.sid,s.epoch,s.public.participants,s.public) for k,s in sessions.values()]
        record(repeat,branch,'all','distinct_delivery_seeds',len(set(seeds))==6)
        for sid,(key,initial) in sessions.items():
            if args.wait_for_models:
                deadline=time.monotonic()+1800
                while True:
                    try:
                        model_status=json.loads((out/'model_metrics.json').read_text())
                        ready=any(r['model']==sid and 'bundle_sha256' in r for r in model_status['rows'])
                    except (FileNotFoundError,json.JSONDecodeError):ready=False
                    if ready:break
                    if time.monotonic()>deadline:raise TimeoutError('model delivery not ready: '+sid)
                    time.sleep(2)
            payload=(out/f'{sid}.zip').read_bytes();expected=sha(payload)
            aad={'sid':sid,'delivery_epoch':initial.epoch,'payload_sha256':expected}
            cipher=encrypt(key,payload,aad);del payload
            if repeat==1 and branch=='rotate':(out/f'{sid}.aesgcm').write_bytes(cipher)
            original_proofs=initial.proofs();original_snapshot=digest(asdict(initial.public))
            context=initial.context();frozen_history={'repeat':repeat,'branch':branch,'sid':sid,
                'cipher_sha256':sha(cipher),'aad':aad,'epochs':[]}
            key0=initial.restore(initial.shares,original_proofs)
            record(repeat,branch,sid,'initial_decryption',matches(key0,cipher,aad,expected))
            snapshots=[initial];state=initial
            for target in range(initial.epoch+1,3):
                contributions=[contribution(state,i) for i in state.public.participants]
                before=state
                bad=copy.deepcopy(contributions)
                # A self-consistent nonzero-constant polynomial must still be refused.
                bad[0]['commitments']=(G,)+bad[0]['commitments'][1:]
                bad[0]['updates']={i:(d+1)%Q for i,d in bad[0]['updates'].items()}
                record(repeat,branch,sid,'reject_nonzero_constant',rejects(lambda:refresh(state,bad)))
                bad_update=copy.deepcopy(contributions);i=state.public.participants[0]
                bad_update[0]['updates'][i]=(bad_update[0]['updates'][i]+1)%Q
                record(repeat,branch,sid,'reject_tampered_update_atomic',rejects(lambda:refresh(state,bad_update)) and state==before)
                record(repeat,branch,sid,'reject_interruption_atomic',rejects(lambda:refresh(state,contributions[:-1])) and state==before)
                refreshed=refresh(state,contributions)
                record(repeat,branch,sid,'legal_zero_refresh',refreshed.epoch==target and refreshed.public.commitments[0]==state.public.commitments[0])
                record(repeat,branch,sid,'refresh_preserves_identity',refreshed.public.identities==state.public.identities)
                refreshed_proofs=refreshed.proofs()
                restored=refreshed.restore(refreshed.shares,refreshed_proofs)
                record(repeat,branch,sid,'refresh_preserves_key',restored==key)
                plain=decrypt(restored,cipher,aad)
                record(repeat,branch,sid,'refresh_preserves_model_hash',sha(plain)==expected)
                with zipfile.ZipFile(io.BytesIO(plain)) as z:
                    # Parse the full safetensors payload, not just ZIP/adapter metadata.
                    from safetensors.torch import load
                    tensors=load(z.read('base/model.safetensors'))
                    adapter=torch.load(io.BytesIO(z.read('adapter.pt')),weights_only=True,map_location='cpu')
                    reload_ok=bool(tensors) and set(adapter)=={'a','b'} and z.testzip() is None
                    del tensors,adapter
                del plain
                record(repeat,branch,sid,'refreshed_model_reload',reload_ok)
                record(repeat,branch,sid,'old_proof_replay_rejected',not any(verify_proof(refreshed.public,p,refreshed.context()) for p in original_proofs))
                # Separate rotation: no share values/commitments change during this step.
                rotated=rotate_session_identity(refreshed,identities[target])
                record(repeat,branch,sid,'identity_rotation_independent',rotated.public.commitments==refreshed.public.commitments and
                       [s.share.value for s in rotated.shares]==[s.share.value for s in refreshed.shares])
                changed=rotated.public.identities!=refreshed.public.identities
                record(repeat,branch,sid,'identity_changed' if branch=='rotate' else 'identity_unchanged',changed if branch=='rotate' else not changed)
                record(repeat,branch,sid,'new_context_proof_valid',all(verify_proof(rotated.public,p,rotated.context()) for p in rotated.proofs()))
                # Change only the identity commitments, retaining the exact old share commitments/context.
                wrong_identity=replace(initial.public,identities=rotated.public.identities)
                valid=all(verify_proof(wrong_identity,p,context) for p in original_proofs)
                record(repeat,branch,sid,'old_proof_new_identity_rejected' if branch=='rotate' else 'old_proof_retained_identity_valid',not valid if branch=='rotate' else valid)
                forged=[replace(s,epoch=target,share=replace(s.share,gamma=identities[target][s.share.i-1])) for s in initial.shares]
                record(repeat,branch,sid,'forged_epoch_commitment_rejected',not any(rotated.public.verify(s.share) for s in forged))
                state=rotated;snapshots.append(state)
            record(repeat,branch,sid,'historical_proof_valid',all(verify_proof(initial.public,p,context) for p in original_proofs))
            record(repeat,branch,sid,'historical_snapshot_unchanged',digest(asdict(initial.public))==original_snapshot)
            record(repeat,branch,sid,'captured_complete_old_shares_still_decrypt',matches(interpolate([s.share for s in initial.shares]),cipher,aad,expected))
            # Exhaust all assignments of retained epochs to each distinct participant.
            mixed_total=interface_rejected=direct_key_equal=direct_decrypted=0
            for selection in itertools.product(range(len(snapshots)),repeat=len(initial.shares)):
                if len(set(selection))==1:continue
                mixed=[snapshots[e].shares[j] for j,e in enumerate(selection)];mixed_total+=1
                interface_rejected+=rejects(lambda:state.restore(mixed,original_proofs))
                k=interpolate([s.share for s in mixed]);direct_key_equal+=int(k==key)
                direct_decrypted+=int(matches(k,cipher,aad,expected))
            if mixed_total:
                record(repeat,branch,sid,'mixed_epochs_interface_rejected',interface_rejected==mixed_total,
                       {'total':mixed_total,'rejected':interface_rejected})
                # Observation only; no asserted impossibility of random collision.
                record(repeat,branch,sid,'mixed_epochs_direct_observation',True,
                       {'total':mixed_total,'same_key':direct_key_equal,'decrypted':direct_decrypted,'claim':'observed counts only'})
            other=sessions[('B' if sid.startswith('A') else 'A')+sid[1:]][1]
            record(repeat,branch,sid,'cross_session_shares_rejected',rejects(lambda:other.restore(initial.shares,original_proofs)))
            record(repeat,branch,sid,'cross_session_proofs_rejected',not any(verify_proof(other.public,p,other.context()) for p in original_proofs))
            wrong_context=dict(context,sid=sid+'-tampered')
            wrong_participants=dict(context,participants=[4,1])
            record(repeat,branch,sid,'tampered_old_context_rejected',not any(verify_proof(initial.public,p,wrong_context) for p in original_proofs))
            record(repeat,branch,sid,'tampered_old_participants_rejected',not any(verify_proof(initial.public,p,wrong_participants) for p in original_proofs))
            wrong_ids=list(initial.public.identities);wrong_ids[initial.public.participants[0]-1]=pow(G,scalar(),P)
            wrong_pub=replace(initial.public,identities=tuple(wrong_ids))
            record(repeat,branch,sid,'tampered_old_identity_rejected',not any(verify_proof(wrong_pub,p,context) for p in original_proofs))
            for snap in snapshots:
                frozen_history['epochs'].append({'epoch':snap.epoch,'public':asdict(snap.public),'context':snap.context(),'proofs':snap.proofs()})
            history.append(frozen_history);del cipher
            save()
        # Registration seed verification after E2 identity rotation uses retained E0 identity.
        for rec in registration:
            if repeat==1 and branch=='rotate':
                original=first_sessions[rec['sid']][1]
                record(repeat,branch,rec['sid'],'historical_seed_matches_trained_model',
                    rec['seed']==identity_seed(rec['sid'],rec['epoch'],rec['participants'],original.public))
        print('DONE',repeat,branch,'checks',len(results),'failures',sum(not r['passed'] for r in results),flush=True)
    report={'status':'completed','repeats':5,'branches':['rotate','keep'],'seconds':time.monotonic()-started,
            'checks':results,'history':history,'failed':sum(not r['passed'] for r in results),
            'model_training_scope':'six trained deliveries only; crypto repeats reuse these fixed payload bytes'}
    (out/'crypto_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',len(results),'checks',report['failed'],'failed',flush=True)

if __name__=='__main__':main()
