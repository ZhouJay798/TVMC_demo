"""Export real-model receipts and independent public verification for all releases."""
import copy
import json
from dataclasses import asdict
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from .bridge import Collector, OneShotBox, evidence_for
from .crypto import prove, recover


def audit_releases(lab, adapters, prompts, state, asset_hash, output):
    from verify_evidence import verify
    rows=[]; collector=Collector();t=state['tardos']
    for i,rec in enumerate(state['records']):
        lab.restore(adapters[i])
        endpoint=f'local-smollm2-user{rec.consumer}';nonce=f'audit-{rec.sid}'
        key_context={'sid':rec.sid,'epoch':rec.epoch,'sha256':asset_hash}
        shares=[state['shares'][j-1] for j in rec.participants]
        proofs=[prove(state['public'],s,key_context) for s in shares]
        key_recovered=recover(state['public'],shares,proofs,key_context)==state['key']
        box=OneShotBox(lab.predict)
        receipt,signature=collector.collect(box,prompts,'batch',endpoint,nonce)
        ev=evidence_for(rec,state['words'][i],state['openings'][i],receipt,signature)
        context={'public':asdict(state['public']),'records':[asdict(r) for r in state['records']],
                 'tardos_p':t.p.tolist(),'threshold':t.threshold,
                 'collector_key':collector.public_key.public_bytes(Encoding.Raw,PublicFormat.Raw).hex(),
                 'endpoint':endpoint,'nonce':nonce,'now':state['now'],
                 'key_proofs':proofs,'key_context':key_context}
        verdict=verify(context,ev)
        decoded=[j+1 for j in t.decode(state['words'],receipt['observed'])]
        negatives={}
        for j,wrong in enumerate(state['records']):
            if j==i:continue
            wrong_ev=evidence_for(wrong,state['words'][j],state['openings'][j],receipt,signature)
            negatives[f'wrong_user_{j+1}_deployment_rejected']=not verify(context,wrong_ev)['deployment_layer']
        edited=copy.deepcopy(ev);edited['receipt']['observed'][0]^=1
        negatives['edited_receipt_rejected']=not verify(context,edited)['deployment_layer']
        negatives['wrong_nonce_rejected']=not verify(dict(context,nonce='different-challenge'),ev)['deployment_layer']
        short,sig=collector.collect(OneShotBox(lab.predict),prompts,'single',endpoint,nonce)
        short_ev=evidence_for(rec,state['words'][i],state['openings'][i],short,sig)
        negatives['signed_single_input_rejected']=not verify(context,short_ev)['deployment_layer']
        try:box.query(prompts,'batch');negatives['second_request_rejected']=False
        except ValueError:negatives['second_request_rejected']=True
        directory=output/'public_audit'/f'user_{i+1}';directory.mkdir(parents=True)
        row={'expected_user':i+1,'decoded_users':decoded,'session':rec.sid,
             'registered_consumer':rec.consumer,'key_recovered':bool(key_recovered),
             'verification':verdict,'negative_checks':negatives,
             'passed':decoded==[i+1] and key_recovered and all(verdict.values()) and all(negatives.values()),
             'scope':'registered session/consumer, not real-world identity or proof of who leaked'}
        for name,value in [('public_context.json',context),('public_evidence.json',ev),('verdict.json',row)]:
            (directory/name).write_text(json.dumps(value,indent=2)+'\n')
        rows.append(row);print('Public audit:',json.dumps(row),flush=True)
    lab.restore(adapters[0])
    return rows
