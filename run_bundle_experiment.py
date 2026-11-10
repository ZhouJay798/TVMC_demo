"""3/3 whole-bundle experiment; one training per replica; five fixture crypto trials.

Large outer ciphertext is retained in memory only to fit local disk constraints.
Issuer public key is a local trust anchor, not evidence of real-world identity.
"""
import argparse,copy,gc,io,itertools,json,tempfile,threading,urllib.request,urllib.error,zipfile,secrets
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from cryptography.exceptions import InvalidTag,InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from run_personal_delivery import setup,rejects
from tvmc.bundle_delivery import make_inner,seal_bundle,open_bundle,open_inner,outer_context
from tvmc.crypto import recover,canonical,encrypt,decrypt,scalar,prove
from tvmc.epochs import BoundShare,Session,contribution,refresh,rotate_session_identity,interpolate
from tvmc.personal_delivery import sha,replica_seed
from tvmc.model import ModelLab,HeadLoRA
from tvmc.tardos import Tardos
from tvmc.reconstruction import request_context,reconstruct,verify_receipt
from tvmc.robustness import corpus,natural_triggers,training_cache,embed_natural,margins,aggregate,utility
from tvmc.merged_delivery import compare_merged_scores,save_merged_model
from tvmc.offline_fixture import make_fixture
from tvmc.plaintext_corpus import load_plaintext_corpus
from tvmc.workflow_report import write_workflow_report
from tvmc.trusted_registry import (code_commitment, create_precommit, create_issuance,
    create_precommit_checkpoint, create_checkpoint, verify_checkpoint,
    verify_issuance, verify_recipient_binding, verify_snapshot)
from serve import make_server


def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def trial(data):
    gate,adm,(k,pub,shares,proofs,ctx),keys,records,_=data;issuer=Ed25519PrivateKey.generate()
    entries=[make_inner(gate,adm,r,b'fixture') for r in records]
    outer,_,_=seal_bundle(gate,adm,records,entries,k,issuer)
    untrusted=[];direct=[]
    for n in range(3):
      for indices in itertools.combinations(range(3),n):
        ss=[shares[i] for i in indices];pp=[proofs[i] for i in indices]
        untrusted.append(rejects(lambda:recover(pub,ss,pp,ctx)))
        if ss:
            candidate=interpolate(ss)
            direct.append(rejects(lambda:decrypt(candidate,outer,outer_context(adm))))
    recovered=recover(pub,shares,proofs,ctx)
    decoded,_=open_bundle(gate,adm,records,outer,recovered,issuer.public_key())
    own=[open_inner(decoded[i],keys[i])==b'fixture' for i in range(3)]
    cross=[rejects(lambda i=i,j=j:open_inner(decoded[i],keys[j])) for i in range(3) for j in range(3) if i!=j]
    wrong_key=rejects(lambda:open_bundle(gate,adm,records,outer,k+1,issuer.public_key()))
    altered=outer[:-1]+bytes([outer[-1]^1]);wrong_outer=rejects(lambda:open_bundle(gate,adm,records,altered,k,issuer.public_key()))
    wrong_issuer=rejects(lambda:open_bundle(gate,adm,records,outer,k,Ed25519PrivateKey.generate().public_key()))
    # Even a holder of k cannot change the issuer-signed manifest without its key.
    raw=decrypt(k,outer,outer_context(adm));files={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():files[name]=z.read(name)
    modified=json.loads(files['manifest.json']);modified['entries'][0]['context']['recipient']=2
    files['manifest.json']=canonical(modified);buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_STORED) as z:
        for name,value in files.items():z.writestr(name,value)
    forged=encrypt(k,buffer.getvalue(),outer_context(adm));bad_manifest=rejects(lambda:open_bundle(gate,adm,records,forged,k,issuer.public_key()))
    swapped=copy.deepcopy(records);swapped[0]['public_key_der']=records[1]['public_key_der']
    bad_binding=rejects(lambda:open_bundle(gate,adm,swapped,outer,k,issuer.public_key()))
    return {'full_recovers':recovered==k,'insufficient_interface_rejections':untrusted,'direct_incomplete_decrypt_failed':direct,
            'own_decrypt':own,'wrong_person_rejections':cross,'wrong_session_key_rejected':wrong_key,
            'tampered_outer_rejected':wrong_outer,'wrong_issuer_rejected':wrong_issuer,
            'modified_manifest_rejected':bad_manifest,'public_key_substitution_rejected':bad_binding}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--persist-bundle',action='store_true',help='write encrypted outer bundle to output directory')
    parser.add_argument('--model-path',type=Path,help='local model directory (default: models/SmolLM2-135M)')
    parser.add_argument('--cache',type=Path,help='local Hugging Face cache (default: .cache/huggingface)')
    parser.add_argument('--offline-fixture',action='store_true',help='use a random synthetic model and data for integration only')
    parser.add_argument('--text-corpus',type=Path,help='UTF-8 public plaintext corpus, split into disjoint windows')
    parser.add_argument('--merged-delivery',action='store_true',help='package a merged model instead of a clean base and separate adapter')
    parser.add_argument('--signed-registry',action='store_true',help='simulate signed registration and checkpoint verification')
    args=parser.parse_args();out=args.output;root=Path(__file__).resolve().parent
    if args.offline_fixture and (args.model_path or args.text_corpus):
        parser.error('--offline-fixture cannot be combined with --model-path or --text-corpus')
    fixture_temp=None;fixture_data=None
    if args.offline_fixture:
        fixture_temp=tempfile.TemporaryDirectory(prefix='tvmc-offline-fixture-')
        source=Path(fixture_temp.name)/'model';fixture_data=make_fixture(source)
    else:
        source=args.model_path or root/'models/SmolLM2-135M'
    cache=args.cache or root/'.cache/huggingface'
    weights=source/'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(f'Model weights not found: {weights}. See README for the pinned model; '
                                'use --offline-fixture only for synthetic integration checks.')
    basehash=sha(weights.read_bytes())
    out.mkdir(parents=True,exist_ok=False,mode=0o700)
    first=None;trials=[]
    for repeat in range(1,6):
        data=setup(basehash);result=trial(data);trials.append({'repeat':repeat,**result})
        print('CRYPTO',repeat,json.dumps(result),flush=True)
        if repeat==1:first=data
    dump(out/'crypto_metrics.json',{'trials':trials,'scope':'five independent crypto trials on small fixture payloads'})
    gate,adm,(key,public,shares,proofs,keyctx),keys,recipients,registration=first
    t=Tardos(n=3);seeds=[replica_seed(adm,r) for r in recipients];words=[t.encode(s) for s in seeds];ids=[f'C-E0-U{i}' for i in (1,2,3)]
    issuer=Ed25519PrivateKey.generate();issuer_public=issuer.public_key();issuer_hex=issuer_public.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()
    registration.update(replica_ids=ids,seeds=seeds,words=[w.tolist() for w in words],p=t.p.tolist(),threshold=t.threshold,issuer_public_key=issuer_hex)
    dump(out/'registration_before_training.json',registration)
    precommit=pre_checkpoint=registry_signer=witness_signer=pre_challenge=openings=None
    if args.signed_registry:
        registry_signer=Ed25519PrivateKey.generate();witness_signer=Ed25519PrivateKey.generate()
        precommit,openings=create_precommit(gate,adm,recipients,words,registry_signer,t)
        pre_challenge=secrets.token_hex(32)
        pre_checkpoint=create_precommit_checkpoint(precommit,registry_signer.public_key(),witness_signer,pre_challenge)
        verify_checkpoint(pre_checkpoint,witness_signer.public_key(),precommit,registry_signer.public_key(),pre_challenge)
        dump(out/'registry_precommit.json',precommit);dump(out/'registry_precommit_checkpoint.json',pre_checkpoint)
        # The experiment codebook is already stored locally. The openings are
        # restricted experiment data, not a production secret-management system.
        openings_path=out/'fingerprint_openings.json';dump(openings_path,openings);openings_path.chmod(0o600)
        def pubhex(key):
            return key.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()
        dump(out/'local_demo_trust_anchors.json',{'issuer':issuer_hex,
             'registry':pubhex(registry_signer.public_key()),'witness':pubhex(witness_signer.public_key()),
             'warning':'Local simulation only; these keys are not authenticated independent trust anchors.'})
    lab=ModelLab(str(source),cache,'cpu',2,True)
    data=fixture_data if args.offline_fixture else (load_plaintext_corpus(args.text_corpus) if args.text_corpus else corpus(cache,True,256))
    prompts=natural_triggers(lab,data['triggers'],3);dump(out/'prompts.json',prompts)
    print('Preparing fixed data and frozen model features',flush=True)
    baseline,predictions=utility(lab,data['test']);features=lab.features(prompts);cal=training_cache(lab,data['calibration'])
    metrics={'status':'running','baseline':baseline,'rows':[],'base_sha256':basehash,'training_count':3,'steps':1200,
             'outer_bundle_persistence':'ciphertext saved to outer_bundle.aesgcm' if args.persist_bundle else 'in-memory only; public hashes and signed manifest retained',
             'crypto_scope':'five independent small fixture repeats; full 3-model bundle and model training performed once',
             'model_scope':'synthetic random fixture; not SmolLM or real-model evidence' if args.offline_fixture else 'local model loaded from model-path',
             'corpus_scope':'synthetic fixture' if args.offline_fixture else ('disjoint windows of one public plaintext corpus' if args.text_corpus else 'cached WikiText-2'),
             'text_corpus_sha256':sha(args.text_corpus.read_bytes()) if args.text_corpus else None,
             'delivery_format':'merged_model' if args.merged_delivery else 'base_and_adapter',
             'signed_registry':'local simulation; independent trust and durable anti-rollback not established' if args.signed_registry else 'not_run'}
    def save():dump(out/'model_metrics.json',metrics)
    def box(active,name,strict=False,operator=None):
        selected=prompts[:1] if strict else prompts;protocol='single' if strict else 'batch'
        server=make_server(lambda ps:margins(active,ps),port=0,max_prompts=1 if strict else 768,response_field='score_differences')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();body=canonical({'protocol':protocol,'prompts':selected})
        client=urllib.request.build_opener(urllib.request.ProxyHandler({}));url=f'http://127.0.0.1:{server.server_address[1]}/query'
        def send():return client.open(urllib.request.Request(url,body,{'Content-Type':'application/json'}),timeout=600)
        try:
            with send() as response:first_status=response.status;answer=json.load(response)
            try:
                with send() as response:second=response.status
            except urllib.error.HTTPError as error:second=error.code
        finally:server.shutdown();server.server_close();thread.join()
        values=answer['score_differences']
        if len(values)==768:
            obs=aggregate(values,3);decoded=t.decode(words,obs);correct=[int((obs==w).sum()) for w in words]
            trace_scores=t.scores(words,obs).tolist()
        else:obs=np.full(256,-1);decoded=[];correct=None;trace_scores=None
        result={'name':name,'operator_label':operator,'inputs':len(selected),'returned_scores':len(values),'protocol':protocol,'first_http':first_status,'second_http':second,
                'decoded_replicas':[ids[i] for i in decoded],'decoded_recipients':[i+1 for i in decoded],
                'observed_bits':int((obs>=0).sum()),'correct_bits':correct,
                'tardos_scores':trace_scores,'tardos_threshold':t.threshold,'individual_leaker_proven':False}
        dump(out/(name+'_http.json'),{'request_sha256':sha(body),'response':answer,'result':result})
        return result
    metrics['negative_control']=box(lab,'clean_negative');save()
    entries=[];snapshots=[];reference_values=[]
    for i,(sid,r) in enumerate(zip(ids,recipients)):
        print('TRAIN',sid,flush=True);torch.manual_seed(10101+i);train=embed_natural(lab,features,np.repeat(words[i],3),cal)
        state=lab.snapshot();snapshots.append(state)
        reference_values.append(margins(lab,prompts) if args.merged_delivery else None)
        if args.merged_delivery:
            dump(out/(sid+'_reference_scores.json'),{'score_differences':reference_values[-1].tolist(),
                 'scope':'local pre-merge output; not an authenticated remote model response'})
        if not args.merged_delivery:torch.save(state,out/(sid+'.pt'))
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_STORED) as z:
            if args.merged_delivery:
                with tempfile.TemporaryDirectory(prefix='merged-',dir=out) as directory:
                    save_merged_model(lab,directory)
                    for path in sorted(Path(directory).iterdir()):
                        if path.is_file():z.write(path,'base/'+path.name)
                z.writestr('delivery_format.json',json.dumps({'format':'merged_model/v1'}))
            else:
                for path in sorted(source.iterdir()):
                    if path.is_file():z.write(path,'base/'+path.name)
                z.writestr('adapter.pt',torch_bytes(state))
            z.writestr('registration.json',json.dumps({'replica':sid,'seed':seeds[i],'recipient':r}))
        payload=buf.getvalue();buf.close();entry=make_inner(gate,adm,r,payload);entries.append(entry)
        metrics['rows'].append({'replica':sid,'recipient':i+1,'training':train,'payload_sha256':sha(payload),'cipher_sha256':sha(entry['ciphertext'])});del payload;save()
    print('Encrypting actual whole bundle',flush=True);outer,manifest,signature=seal_bundle(gate,adm,recipients,entries,key,issuer);del entries;gc.collect()
    if args.persist_bundle:(out/'outer_bundle.aesgcm').write_bytes(outer)
    metrics['outer_cipher_sha256']=sha(outer);metrics['outer_cipher_bytes']=len(outer);dump(out/'signed_manifest.json',{'manifest':manifest,'signature_hex':signature.hex(),'issuer_public_key':issuer_hex});save()
    issued=checkpoint=checkpoint_challenge=None
    if args.signed_registry:
        issued=create_issuance(precommit,registry_signer.public_key(),manifest,signature,issuer_public,
                               gate,adm,recipients,registry_signer)
        checkpoint_challenge=secrets.token_hex(32)
        checkpoint=create_checkpoint(precommit,issued,registry_signer.public_key(),witness_signer,
                                     checkpoint_challenge,pre_checkpoint,pre_challenge)
        issued_body=verify_issuance(precommit,issued,registry_signer.public_key(),manifest,signature,issuer_public,
                                    gate,adm,recipients)
        verify_checkpoint(checkpoint,witness_signer.public_key(),issued,registry_signer.public_key(),checkpoint_challenge)
        tampered=copy.deepcopy(issued);tampered['body']['entries'][0]['recipient']=2
        changed_manifest=copy.deepcopy(manifest);changed_manifest['entries'][0]['context']['recipient']=2
        checks={
            'issued_signature_and_binding_valid':True,
            'tampered_registry_rejected':rejects(lambda:verify_snapshot(tampered,registry_signer.public_key())),
            'tampered_issuer_manifest_rejected':rejects(lambda:verify_issuance(precommit,issued,registry_signer.public_key(),changed_manifest,signature,issuer_public,gate,adm,recipients)),
            'older_snapshot_rejected_by_current_checkpoint':rejects(lambda:verify_checkpoint(checkpoint,witness_signer.public_key(),precommit,registry_signer.public_key(),checkpoint_challenge)),
            'wrong_checkpoint_challenge_rejected':rejects(lambda:verify_checkpoint(checkpoint,witness_signer.public_key(),issued,registry_signer.public_key(),'0'*64)),
            'expired_checkpoint_rejected':rejects(lambda:verify_checkpoint(checkpoint,witness_signer.public_key(),issued,registry_signer.public_key(),checkpoint_challenge,now=checkpoint['body']['expires_at'])),
            'wrong_recipient_binding_rejected':rejects(lambda:verify_recipient_binding(issued_body,ids[1],1,bytes.fromhex(recipients[0]['public_key_der']))),
        }
        dump(out/'registry_issued.json',issued);dump(out/'registry_checkpoint.json',checkpoint)
        dump(out/'registry_checks.json',{'checks':checks,'scope':'local issuer, registrar and witness simulation; no independent trust or durable latest-state guarantee'})
        metrics['registry_checks']=checks;save()
    receipt_context=request_context(public,keyctx,sha(outer),manifest,secrets.token_hex(32))
    # In this local simulation each participant creates a request-bound proof.
    submitted_proofs=[prove(public,share,receipt_context) for share in shares]
    restored,receipt=reconstruct(public,shares,submitted_proofs,receipt_context,issuer)
    dump(out/'reconstruction_receipt.json',receipt)
    metrics['reconstruction_receipt']=verify_receipt(receipt,public,receipt_context,issuer_public)
    metrics['session_key_restored']=restored==key
    entries,verified_manifest=open_bundle(gate,adm,recipients,outer,restored,issuer_public)
    if args.signed_registry:
        # Revalidate immediately before releasing any individual plaintext.
        issued_body=verify_issuance(precommit,issued,registry_signer.public_key(),verified_manifest,signature,issuer_public,
                                    gate,adm,recipients)
        verify_checkpoint(checkpoint,witness_signer.public_key(),issued,registry_signer.public_key(),checkpoint_challenge)
    metrics['bundle_signature_valid']=True;metrics['wrong_session_key_rejected']=rejects(lambda:open_bundle(gate,adm,recipients,outer,key+1,issuer_public));gc.collect()
    for i,entry in enumerate(entries):
        sid=ids[i];row=metrics['rows'][i];row['cross_key_rejected']={str(j+1):rejects(lambda j=j:open_inner(entry,keys[j])) for j in range(3) if i!=j}
        if args.signed_registry:
            registered=verify_recipient_binding(issued_body,sid,i+1,bytes.fromhex(recipients[i]['public_key_der']))
            if registered['code_commitment']!=code_commitment(sid,words[i],openings[sid]):
                raise ValueError('registered fingerprint commitment mismatch')
            row['registered_binding_and_code_commitment_valid']=True
        plaintext=open_inner(entry,keys[i]);row['owner_decrypt_hash_matches']=sha(plaintext)==row['payload_sha256']
        with tempfile.TemporaryDirectory(dir=out) as temp:
            temp=Path(temp)
            with zipfile.ZipFile(io.BytesIO(plaintext)) as z:z.extractall(temp)
            del plaintext
            active=ModelLab(str(temp/'base'),cache,'cpu',2,True)
            if not args.merged_delivery:active.restore(torch.load(temp/'adapter.pt',weights_only=True,map_location='cpu'))
            batch=box(active,sid+'_batch');single=box(active,sid+'_strict',strict=True);u,_=utility(active,data['test'],predictions)
            row.update(batch=batch,strict_single=single,ppl=u['perplexity'],ppl_increase_pct=100*(u['perplexity']/baseline['perplexity']-1))
            if args.merged_delivery:
                with (out/(sid+'_batch_http.json')).open(encoding='utf-8') as f:observed=json.load(f)['response']['score_differences']
                comparison=compare_merged_scores(reference_values[i],observed)
                row['merged_reload_comparison']=comparison
                row['merged_reload_max_abs_score_difference']=comparison['max_abs_score_difference']
                row['merged_reload_scores_match']=comparison['accepted']
                row['contains_separate_adapter']=(temp/'adapter.pt').exists()
            if i==0:metrics['transfer_control']=box(active,'U1_copy_operator_U2',operator=2)
            print('MODEL',json.dumps(row),flush=True);del active
        save();gc.collect()
    # Refresh existing 3/3 shares, then independently rotate identities; the old
    # issuer-signed bundle remains tied to its historical identity snapshot.
    old=Session(adm.sid,0,public,tuple(BoundShare(adm.sid,0,s) for s in shares))
    new=refresh(old,[contribution(old,i) for i in adm.participants]);fresh=[scalar() for _ in range(5)];rotated=rotate_session_identity(new,fresh)
    restored2=rotated.restore(rotated.shares,rotated.proofs())
    reopened,_=open_bundle(gate,adm,recipients,outer,restored2,issuer_public)
    outer_reopened=all(sha(e['ciphertext'])==sha(entries[i]['ciphertext']) for i,e in enumerate(reopened))
    del reopened,outer;gc.collect()
    metrics['combined_refresh_rotation']={'full_outer_bundle_reopened_after_refresh':outer_reopened,'share_values_changed':[a.share.value!=b.share.value for a,b in zip(old.shares,rotated.shares)],
       'identity_commitments_changed':[old.public.identities[i-1]!=rotated.public.identities[i-1] for i in adm.participants],
       'session_key_unchanged':restored2==key,'historical_bundle_keys_still_usable':[sha(open_inner(e,keys[i]))==metrics['rows'][i]['payload_sha256'] for i,e in enumerate(entries)],
       'note':'Refreshed and identity-rotated shares recovered the same key and reopened the original complete outer ciphertext before memory release.'}
    # Real model-parameter mixing, not simulated bit interleaving. The rank-16
    # update gives exactly 0.5*(B1@A1/8+B2@A2/8).
    mixed=[]
    for i,j in itertools.combinations(range(3),2):
        head=HeadLoRA(lab.head.base,lab.token_ids,rank=16)
        with torch.no_grad():head.a.copy_(torch.cat([snapshots[i]['a'],snapshots[j]['a']],0));head.b.copy_(torch.cat([snapshots[i]['b'],snapshots[j]['b']],1))
        lab.head=head;lab.model.lm_head=head
        result=box(lab,f'mixed_U{i+1}_U{j+1}');hit=set(result['decoded_replicas']) & {ids[i],ids[j]};wrong=set(result['decoded_replicas'])-{ids[i],ids[j]}
        mixed.append({'pair':[ids[i],ids[j]],'decoded':result['decoded_replicas'],'at_least_one_source_hit':bool(hit),'innocent_hits':sorted(wrong)})
        print('MIXED',json.dumps(mixed[-1]),flush=True)
    metrics['real_model_mixtures']=mixed
    checks={'crypto_fixtures_passed':all(all(v) if isinstance(v,list) else v for trow in trials for k,v in trow.items() if k!='repeat'),
            'clean_negative_has_no_hit':not metrics['negative_control']['decoded_replicas'],
            'all_original_copies_identified':all(row['batch']['decoded_replicas']==[row['replica']] for row in metrics['rows']),
            'all_own_decryptions_and_cross_key_rejections':all(row['owner_decrypt_hash_matches'] and all(row['cross_key_rejected'].values()) for row in metrics['rows']),
            'all_second_requests_rejected':all(row['batch']['second_http']==409 and row['strict_single']['second_http']==409 for row in metrics['rows']),
            'strict_single_not_attributed':all(not row['strict_single']['decoded_replicas'] for row in metrics['rows'])}
    if args.merged_delivery:
        checks['merged_models_reload_equivalently']=all(row['merged_reload_scores_match'] and not row['contains_separate_adapter'] for row in metrics['rows'])
    if args.signed_registry:checks['all_registry_controls_passed']=all(metrics['registry_checks'].values())
    metrics['workflow_checks']=checks
    metrics['status']='completed' if all(checks.values()) else 'completed_with_failed_checks'
    save();write_workflow_report(out/'WORKFLOW_REPORT.md',metrics)
    print('COMPLETE',json.dumps({'status':metrics['status'],'workflow_checks':checks},ensure_ascii=False),flush=True)
    if fixture_temp is not None:fixture_temp.cleanup()
    return 0 if all(checks.values()) else 2


def torch_bytes(state):
    buffer=io.BytesIO();torch.save(state,buffer);return buffer.getvalue()

if __name__=='__main__':raise SystemExit(main())
