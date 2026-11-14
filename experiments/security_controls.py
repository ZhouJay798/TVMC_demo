"""Isolated defense-on/off experiments; never disables production safeguards."""
import sys,json,copy,csv,secrets
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from tvmc.paper_workflow import Custody,sign,verify,sha,Ledger
from tvmc.pki import DemoCA,Trust,Registry,pem,member_key,issuer_key
from tvmc import paper_hpke as hpke,paper_threshold as te
from tvmc.crypto import digest,canonical
from tvmc.paper_trace import FileTracer
from tvmc.score_approval import ScoreApprover,score_ciphertext


def accepted(call):
    try:
        result=call()
        return (bool(result),'accepted' if result else 'rejected')
    except (ValueError,KeyError,TypeError) as exc:return False,str(exc)


def run(output,rounds=5):
    if output.exists():raise ValueError("Use a new output directory; retain previous evidence")
    rows=[]
    def record(trial,case,level,off,on,boundary=False,normal=False):
        a,ar=accepted(off);b,br=accepted(on)
        rows.append(dict(trial=trial,case=case,impact=level,without_control=a,with_control=b,
                         off_reason=ar,on_reason=br,known_boundary=boundary,normal_control=normal))
        output.mkdir(parents=True,exist_ok=True)
        (output/'trials.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
        print(trial,case,'off=',a,'on=',b,flush=True)
    for trial in range(rounds):
        c=Custody();evil=Ed25519PrivateKey.generate();rx=X25519PrivateKey.generate()
        message={'member':1,'request':'release-copy','nonce':secrets.token_hex(16)}
        genuine=sign(c.members[0].identity,'request',message)
        record(trial,'normal_identity','normal_request',
            lambda:verify(bytes.fromhex(c.registry[1]['identity']),'request',message,genuine),
            lambda:verify(member_key(c.registry,1),'request',message,genuine),normal=True)
        normalkey=secrets.token_bytes(32)
        def normal_receive(guarded):
            pk=member_key(c.registry,1,'receiving') if guarded else bytes.fromhex(c.registry[1]['receiving'])
            return hpke.open(c.members[0].receiving,hpke.seal(pk,normalkey,b'normal'),b'normal')==normalkey
        record(trial,'normal_receiving','normal_data_key_delivery',lambda:normal_receive(False),lambda:normal_receive(True),normal=True)
        normal_sig=sign(c.issuer,'manifest',message)
        record(trial,'normal_issuer','normal_manifest',
            lambda:verify(c.issuer.public_key().public_bytes_raw(),'manifest',message,normal_sig),
            lambda:verify(issuer_key(c.registry),'manifest',message,normal_sig),normal=True)
        sig=sign(evil,'request',message)
        bad=copy.deepcopy(c.registry);bad[1]['identity']=evil.public_key().public_bytes_raw().hex()
        record(trial,'identity_key_swap','forged_request_accepted',
            lambda:verify(bytes.fromhex(bad[1]['identity']),'request',message,sig),
            lambda:verify(member_key(bad,1),'request',message,sig))
        badrx=copy.deepcopy(c.registry);badrx[1]['receiving']=rx.public_key().public_bytes_raw().hex()
        key=secrets.token_bytes(32)
        def recover(reg,guarded):
            pk=member_key(reg,1,'receiving') if guarded else bytes.fromhex(reg[1]['receiving'])
            return hpke.open(rx,hpke.seal(pk,key,b'copy-1'),b'copy-1')==key
        record(trial,'receiving_key_swap','attacker_recovers_data_key',lambda:recover(badrx,False),lambda:recover(badrx,True))
        forged=sign(evil,'manifest',message)
        record(trial,'issuer_key_swap','forged_manifest_accepted',
            lambda:verify(evil.public_key().public_bytes_raw(),'manifest',message,forged),
            lambda:verify(issuer_key(c.registry,evil.public_key().public_bytes_raw()),'manifest',message,forged))
        other=copy.deepcopy(c.registry);other[1]=copy.deepcopy(c.registry[2]);other[1]['i']=1
        sig2=sign(c.members[1].identity,'request',message)
        record(trial,'other_member_certificate','wrong_identity_accepted',
            lambda:verify(bytes.fromhex(other[1]['identity']),'request',message,sig2),
            lambda:verify(member_key(other,1),'request',message,sig2))
        fakeca=DemoCA();fakecert=fakeca.issue('U1',evil.public_key(),'identity')
        foreign=copy.deepcopy(c.registry);foreign[1]['identity']=evil.public_key().public_bytes_raw().hex();foreign[1]['identity_certificate']=fakecert
        attacker_trust=Trust(pem(fakeca.root),fakeca.crl(),fakeca.issue('distributor',evil.public_key(),'issuer'))
        replaced=Registry(attacker_trust);replaced.update(copy.deepcopy(foreign))
        record(trial,'root_supplied_by_attacker','forged_certificate_accepted',
            lambda:verify(member_key(replaced,1),'request',message,sig),
            lambda:verify(member_key(foreign,1),'request',message,sig))
        # Deliberately show that control of the verifier's trust root defeats CA.
        record(trial,'verifier_trust_root_compromised','trust_boundary',
            lambda:verify(member_key(replaced,1),'request',message,sig),
            lambda:verify(member_key(replaced,1),'request',message,sig),True)
        savedcrl=c.registry.trust.crl_pem
        revoked=copy.deepcopy(c.registry);revoked.trust.update_crl(c.ca.revoke(revoked[1]['receiving_certificate']))
        record(trial,'revoked_receiving_certificate','revoked_key_accepted',
            lambda:bytes.fromhex(revoked[1]['receiving']),lambda:member_key(revoked,1,'receiving'))
        rollback=copy.deepcopy(revoked);rollback.trust.crl_pem=savedcrl
        weak=copy.deepcopy(rollback);weak.trust.min_crl_number=0
        record(trial,'old_crl_rollback','revoked_key_accepted',lambda:member_key(weak,1,'receiving'),lambda:member_key(rollback,1,'receiving'))
        # Return CA revocation state only affects new CRLs, not c's original CRL.
        p1=c.issue(1,b'model-one','copy-one');p2=c.issue(2,b'model-two','copy-two')
        tampered=copy.deepcopy(p1);tampered['manifest']['message']['recipient']=2
        record(trial,'manifest_recipient_tamper','bad_manifest_accepted',
            lambda:tampered['manifest']['message']['recipient']==2,
            lambda:c.validate_package(tampered))
        record(trial,'valid_U2_package_given_to_U1','wrong_recipient_package_accepted',
            lambda:verify(c.issuer_pk,'manifest',p2['manifest']['message'],p2['manifest']['signature']),
            lambda:c.request(c.members[0],p2))
        changed=copy.deepcopy(p1);ct=bytes.fromhex(changed['chunks'][0]);changed['chunks'][0]=(bytes([ct[0]^1])+ct[1:]).hex()
        record(trial,'ciphertext_tamper','bad_package_accepted_not_plaintext_recovered',
            lambda:verify(c.issuer_pk,'manifest',changed['manifest']['message'],changed['manifest']['signature']),
            lambda:c.validate_package(changed))
        old=copy.deepcopy(c.ledger);oldhead=old.entries[-1]['hash'];c.ledger.append('new_checkpoint',{'trial':trial})
        record(trial,'valid_old_log_rollback','old_history_accepted',lambda:old.check(),lambda:old.check(c.ledger.entries[-1]['hash']))
        altered=copy.deepcopy(c.ledger);altered.entries[0]['body']['registration']['identity']='00'*32
        prev='0'*64
        for e in altered.entries:
            e['prev']=prev;e['hash']=digest({k:v for k,v in e.items() if k!='hash'});prev=e['hash']
        record(trial,'rewrite_log_and_rehash','forged_history_accepted',lambda:altered.check(),lambda:altered.check(c.ledger.entries[-1]['hash']))
        # Fresh group: encryption of a bit must not be accepted as a score ciphertext.
        t=FileTracer(c,[.5]*4,2,'guard-'+str(trial));word=[1,0,1,0];cid='guard-copy'
        t.enroll(cid,word);artifact=json.dumps(word).encode();c.issue(1,artifact,cid);t.commit_file(artifact)
        c.ledger.append('group_disclosed_and_retired',dict(parameters=t.params,salt=t.salt))
        req=dict(group=t.group,kid=c.code_key.kid,head=c.ledger.entries[-1]['hash'],evidence=sha(artifact),observed=word,
                 scores=[dict(cid=cid,a=t.codebook[cid][0]['a'],b=t.codebook[cid][0]['b'])])
        c.ledger.append('score_request',req);head=c.ledger.entries[-1]['hash']
        def unsafe_bit_recovery():
            ct=t.codebook[cid][0];ctx='unchecked-ciphertext'
            proofs=[te.partial(c.code_key,s,ct['a'],ctx) for s in c.code_shares[:3]]
            shared=te.combine(c.code_key,ct['a'],proofs,ctx)
            return ct['b']*pow(shared,-1,te.P)%te.P==te.G
        guard=ScoreApprover(c.code_key,c.code_shares[0],c.members[0].identity,c.issuer_pk,c.registry,json.loads)
        record(trial,'fingerprint_bit_disguised_as_score','fingerprint_bit_disclosure',unsafe_bit_recovery,
            lambda:guard.approve(req,c.ledger.entries,head,artifact))
        # Threshold is a mathematical barrier; no policy ablation can make two valid shares three.
        cap=te.seal(c.release_key,b'secret',b'label'.hex());ctx='threshold'
        proofs=[te.partial(c.release_key,s,cap['a'],ctx) for s in c.release_shares[:2]]
        record(trial,'two_shares_only','mathematical_threshold_control',
            lambda:te.open_capsule(c.release_key,cap,proofs,b'label'.hex(),ctx),
            lambda:te.open_capsule(c.release_key,cap,proofs,b'label'.hex(),ctx),True)
        ctx='offline-collusion';ct=t.codebook[cid][0]
        def collusion():return unsafe_bit_recovery()
        record(trial,'three_members_offline_collusion','threshold_collusion_boundary',collusion,collusion,True)
    summary=[]
    for case in dict.fromkeys(r['case'] for r in rows):
        subset=[r for r in rows if r['case']==case]
        summary.append(dict(case=case,rounds=len(subset),without_control_accepted=sum(r['without_control'] for r in subset),
            with_control_accepted=sum(r['with_control'] for r in subset),impact=subset[0]['impact'],known_boundary=subset[0]['known_boundary'],normal_control=subset[0]['normal_control']))
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
    assert all(r['with_control'] and r['without_control'] for r in rows if r['normal_control'])
    assert all(not r['with_control'] for r in rows if not r['known_boundary'] and not r['normal_control'])
    assert all(r['without_control'] for r in rows if not r['known_boundary'] and not r['normal_control'])

if __name__=='__main__':run(Path(sys.argv[1]),int(sys.argv[2]) if len(sys.argv)>2 else 5)
