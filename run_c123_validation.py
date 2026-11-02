"""Additional C1/C2/C3 controls on a completed end-to-end bundle run.

No retraining: transformations operate on the newly trained adapters. Every
probe creates its own local one-shot HTTP endpoint. No endpoint identity proof
is supplied. Text control returns the true unrestricted greedy next token.
"""
import argparse,copy,json,threading,urllib.request,urllib.error
from pathlib import Path
import numpy as np
import torch
from torch import nn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from tvmc.crypto import canonical
from tvmc.epochs import EpochPublic
from tvmc.reconstruction import request_context,verify_receipt
from tvmc.model import ModelLab,HeadLoRA
from tvmc.robustness import corpus,margins,aggregate,utility,group_quantize
from tvmc.tardos import Tardos
from run_personal_delivery import rejects
from serve import make_server


def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

@torch.no_grad()
def greedy_tokens(lab,prompts):
    result=[]
    for start in range(0,len(prompts),16):
        inputs=lab._tokenize(prompts[start:start+16],40)
        logits=lab.logits(inputs);last=inputs['attention_mask'].sum(1)-1
        ids=logits[torch.arange(len(last)),last].argmax(-1).tolist()
        result.extend(lab.tokenizer.decode([i]) for i in ids)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();out=a.run_dir
    reg=json.loads((out/'registration_before_training.json').read_text());m=json.loads((out/'model_metrics.json').read_text())
    if m['status']!='completed':raise ValueError('complete full delivery first')
    manifest=json.loads((out/'signed_manifest.json').read_text())['manifest'];receipt=json.loads((out/'reconstruction_receipt.json').read_text())
    pub=EpochPublic(**reg['key_public']);signer=Ed25519PublicKey.from_public_bytes(bytes.fromhex(reg['issuer_public_key']))
    ctx=request_context(pub,reg['key_context'],m['outer_cipher_sha256'],manifest,receipt['record']['context']['request_nonce'])
    checked=verify_receipt(receipt,pub,ctx,signer);seen=set();verify_receipt(receipt,pub,ctx,signer,seen)
    controls={'valid_receipt':True,'replay_rejected':rejects(lambda:verify_receipt(receipt,pub,ctx,signer,seen))}
    for field in ('request_nonce','bundle_hash','manifest_hash','session'):
        changed=copy.deepcopy(ctx);changed[field]='tampered'
        controls[field+'_substitution_rejected']=rejects(lambda:verify_receipt(receipt,pub,changed,signer))
    changed=copy.deepcopy(receipt);changed['record']['participants']=[1]
    controls['participant_tamper_rejected']=rejects(lambda:verify_receipt(changed,pub,ctx,signer))
    report={'C1':{'controls':controls,'verified_participants':checked['verified_participants'],
        'limitation':'Trusted service attestation, not proof of the human seller or all offline reconstructions'},'C2':[],'C3':[]}
    root=Path(__file__).resolve().parent;lab=ModelLab(str(root/'models/SmolLM2-135M'),root/'.cache/huggingface','cpu',2,True)
    data=corpus(root/'.cache/huggingface',True,256);prompts=json.loads((out/'prompts.json').read_text());t=Tardos(n=3);words=reg['words']
    base_ppl=m['baseline']['perplexity'];base=lab.head.base
    def save():dump(out/'c123_metrics.json',report)
    def probe(name,text=False):
        field='output_tokens' if text else 'score_differences'
        server=make_server(lambda ps:greedy_tokens(lab,ps) if text else margins(lab,ps),port=0,max_prompts=768,response_field=field)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        body=canonical({'protocol':'batch','prompts':prompts});client=urllib.request.build_opener(urllib.request.ProxyHandler({}));url=f'http://127.0.0.1:{server.server_address[1]}/query'
        def send():return client.open(urllib.request.Request(url,body,{'Content-Type':'application/json'}),timeout=600)
        try:
            with send() as r:status=r.status;response=json.load(r)
            try:
                with send() as r:second=r.status
            except urllib.error.HTTPError as e:second=e.code
        finally:server.shutdown();server.server_close();worker.join()
        values=response[field]
        if text:
            raw=np.array([int(x.strip()) if x.strip() in ('0','1') else -1 for x in values]).reshape(256,3)
            bits=np.full(256,-1,dtype=int)
            for k,row in enumerate(raw):
                if (row==0).sum()>=2:bits[k]=0
                elif (row==1).sum()>=2:bits[k]=1
        else:bits=aggregate(values,3)
        decoded=t.decode(words,bits)
        row={'name':name,'first_http':status,'second_http':second,'proof_supplied':False,
             'observed_bits':int((bits>=0).sum()),'correct_bits':[int((bits==w).sum()) for w in np.array(words)],
             'decoded_replicas':[reg['replica_ids'][i] for i in decoded],'registered_custodians':[reg['recipients'][i]['recipient'] for i in decoded],
             'scores':t.scores(words,bits).tolist(),'threshold':t.threshold,'individual_leaker_proven':False}
        dump(out/(name+'_http.json'),{'response':response,'result':row})
        return row
    for i in range(3):
        state=torch.load(out/f'C-E0-U{i+1}.pt',weights_only=True,map_location='cpu')
        lab.head=HeadLoRA(base,lab.token_ids);lab.model.lm_head=lab.head;lab.restore(state)
        row=probe(f'C2_U{i+1}_text_only',text=True);report['C2'].append(row);print('TEXT',json.dumps(row),flush=True);save()
        delta=(state['b']@state['a'])/8
        # The merge clones the output head: tied token embeddings are not changed.
        for transform in ('merged_head','int8_delta','prune50_delta','remove_delta'):
            change=delta.clone()
            if transform=='int8_delta':change=group_quantize(change)
            elif transform=='prune50_delta':
                flat=change.abs().flatten();indices=torch.argsort(flat)[:flat.numel()//2];change.flatten()[indices]=0
            elif transform=='remove_delta':change.zero_()
            merged=nn.Linear(base.in_features,base.out_features,bias=False)
            with torch.no_grad():merged.weight.copy_(base.weight);merged.weight[lab.token_ids]+=change
            lab.head=HeadLoRA(merged,lab.token_ids);lab.model.lm_head=lab.head
            row=probe(f'C3_U{i+1}_{transform}');u,_=utility(lab,data['test'])
            row.update(transform=transform,source_custodian=i+1,ppl=u['perplexity'],ppl_increase_pct=100*(u['perplexity']/base_ppl-1),
                       exact_source_match=row['registered_custodians']==[i+1],
                       wrong_hits=[j for j in row['registered_custodians'] if j!=i+1])
            report['C3'].append(row);print('TRANSFORM',json.dumps(row),flush=True);save()
    report['status']='completed';report['training_repeats_per_replica']=1;save()

if __name__=='__main__':main()
