"""Research controls: real calibration, reversible compression, and held-out utility.

sparsegpt_prune adapts the SparseGPT blockwise second-order reconstruction
algorithm (IST-DASLab/sparsegpt, Apache-2.0). See third_party attribution.
No private trigger data is used to calibrate compression or train students.
"""
import hashlib
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn

DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
MODEL_REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'


def corpus(cache, offline=False, test_offset=128):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    seen, splits = set(), {}
    for split in ('train', 'validation', 'test'):
        path = hf_hub_download('Salesforce/wikitext',
                  f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet', repo_type='dataset',
                  revision=DATA_REVISION, cache_dir=str(cache), local_files_only=offline)
        rows = []
        for text in pq.read_table(path, columns=['text'])['text'].to_pylist():
            text = text.strip()
            key = hashlib.sha256(text.encode()).hexdigest()
            if len(text) >= 240 and not text.startswith('=') and key not in seen:
                seen.add(key)
                rows.append(text)
        splits[split] = rows
    # Nonoverlapping documents. These allocations are fixed before measuring tests.
    # Spread calibration across articles, excluding all trigger/student rows.
    calibration_ids=np.random.default_rng(2026).choice(np.arange(384,len(splits['train'])),128,replace=False)
    allocation = {'calibration': [splits['train'][int(i)] for i in calibration_ids], 'distillation': splits['train'][64:128],
                  'triggers': splits['train'][128:384], 'validation': splits['validation'][:64],
                  'test': splits['test'][test_offset:test_offset+64]}
    if any(len(v) != (256 if k == 'triggers' else 128 if k=='calibration' else 64) for k,v in allocation.items()):
        raise ValueError('pinned corpus does not contain the required disjoint documents')
    return allocation


def batches(lab, texts, size=4, length=96):
    for offset in range(0,len(texts),size):
        yield lab._tokenize(texts[offset:offset+size], length)


@torch.no_grad()
def utility(lab, texts, baseline=None):
    loss, count, same = 0., 0, 0
    predictions = []
    for index, inputs in enumerate(batches(lab,texts)):
        logits=lab.logits(inputs)[:, :-1]
        labels=inputs['input_ids'][:, 1:]
        mask=inputs['attention_mask'][:, 1:].bool()
        loss += float(nn.functional.cross_entropy(logits[mask], labels[mask], reduction='sum'))
        count += int(mask.sum())
        pred=logits.argmax(-1)[mask].cpu()
        predictions.append(pred)
        if baseline is not None:
            same += int((pred==baseline[index]).sum())
    nll=loss/count
    return {'nll':nll,'perplexity':float(np.exp(nll)), 'tokens':count,
            'next_token_agreement':same/count if baseline is not None else 1.}, predictions


def utility_result(measured, baseline):
    increase=measured['perplexity']/baseline['perplexity']-1
    return dict(measured, relative_perplexity_increase=increase,
                utility_pass=increase<=.10 and measured['next_token_agreement']>=.90)


@torch.no_grad()
def hessians(lab, texts):
    grams, counts, handles = {}, {}, []
    def hook(name):
        def collect(module, inputs):
            x=inputs[0].detach().float().reshape(-1,module.in_features)
            grams[name].add_(x.T@x)
            counts[name]+=len(x)
        return collect
    for name,module in lab.model.model.named_modules():
        if isinstance(module, nn.Linear):
            grams[name]=torch.zeros(module.in_features,module.in_features,device=lab.device)
            counts[name]=0
            handles.append(module.register_forward_pre_hook(hook(name)))
    try:
        # Full fixed-size blocks contain no pad activations in the long corpus.
        for inputs in batches(lab,texts,size=4,length=64):
            if not bool(inputs['attention_mask'].all()):
                # Feed each short row without padding to exclude fake calibration tokens.
                for i in range(len(inputs['input_ids'])):
                    n=int(inputs['attention_mask'][i].sum())
                    lab.model.model(input_ids=inputs['input_ids'][i:i+1,:n],use_cache=False)
            else:
                lab.model.model(**inputs,use_cache=False)
    finally:
        for handle in handles: handle.remove()
    return {k:v/max(counts[k],1) for k,v in grams.items()}


@torch.no_grad()
def group_quantize(weight, group_size=32):
    if weight.ndim!=2 or weight.shape[1]%group_size:
        raise ValueError('group size must divide input dimension')
    groups=weight.reshape(weight.shape[0],-1,group_size)
    scale=groups.abs().amax(-1,keepdim=True).clamp_min(1e-8)/127
    return ((groups/scale).round().clamp(-127,127)*scale).reshape_as(weight)


@torch.no_grad()
def sparsegpt_prune(weight, gram, sparsity=.30, block_size=128, damping=.01):
    """Blockwise pruning + inverse-Hessian error compensation; float inference.

    Adapted from SparseGPT fasterprune; CPU-safe, exact per-block mask count,
    input validation, caller-owned weights, and no quantization coupling.
    """
    if not 0<=sparsity<1 or damping<=0 or block_size<1:
        raise ValueError('invalid pruning parameters')
    w=weight.float().clone()
    h=gram.float().clone()
    if h.shape != (w.shape[1],w.shape[1]): raise ValueError('Hessian shape mismatch')
    dead=h.diag()==0
    h[dead,dead]=1
    w[:,dead]=0
    h.diagonal().add_(damping*h.diag().mean())
    inverse=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)),upper=True)
    for start in range(0,w.shape[1],block_size):
        end=min(start+block_size,w.shape[1])
        local=w[:,start:end].clone()
        block=inverse[start:end,start:end]
        importance=local.square()/block.diag().square()[None,:]
        k=round(importance.numel()*sparsity)
        mask=torch.zeros_like(importance,dtype=torch.bool)
        if k: mask.flatten()[importance.flatten().topk(k,largest=False).indices]=True
        errors=torch.zeros_like(local)
        out=torch.zeros_like(local)
        for j in range(end-start):
            column=local[:,j].clone()
            kept=column.masked_fill(mask[:,j],0)
            error=(column-kept)/block[j,j]
            out[:,j]=kept
            local[:,j:]-=error[:,None]*block[j,j:][None,:]
            errors[:,j]=error
        w[:,start:end]=out
        w[:,end:]-=errors@inverse[start:end,end:]
    return w.to(weight.dtype)


@contextmanager
def compressed(lab, method, grams=None):
    originals=[]
    total=zeros=0
    try:
        with torch.no_grad():
            for name,module in lab.model.model.named_modules():
                if not isinstance(module,nn.Linear): continue
                weight=module.weight
                originals.append((weight,weight.detach().cpu().clone()))
                if method=='row_int8':
                    scale=weight.abs().amax(1,keepdim=True).clamp_min(1e-8)/127
                    transformed=(weight/scale).round().clamp(-127,127)*scale
                elif method=='group_int8':
                    transformed=group_quantize(weight)
                elif method=='magnitude30':
                    # Identical threshold convention to the original Demo.
                    transformed=weight*(weight.abs()>torch.quantile(weight.abs().flatten(),.30))
                elif method=='sparsegpt30':
                    if grams is None: raise ValueError('calibration required')
                    transformed=sparsegpt_prune(weight,grams[name])
                elif method=='sparsegpt_mlp30':
                    if grams is None: raise ValueError('calibration required')
                    # SmolLM2's MLP accounts for exactly 75% of transformer
                    # Linear weights: 40% MLP sparsity gives 30% overall.
                    linear_total=sum(m.weight.numel() for _,m in lab.model.model.named_modules() if isinstance(m,nn.Linear))
                    mlp_total=sum(m.weight.numel() for n,m in lab.model.model.named_modules() if isinstance(m,nn.Linear) and '.mlp.' in n)
                    if not mlp_total: raise ValueError('no MLP matrices')
                    fraction=.30*linear_total/mlp_total
                    transformed=sparsegpt_prune(weight,grams[name],fraction) if '.mlp.' in name else weight.clone()
                else: raise ValueError(method)
                weight.copy_(transformed)
                total+=weight.numel()
                zeros+=int((weight==0).sum())
        yield {'method':method,'matrix_weights':total,'zero_weights':zeros,
               'actual_zero_fraction':zeros/total,'scope':'all transformer Linear weights; embeddings/head excluded',
               'execution':'float32 after weight transformation'}
    finally:
        with torch.no_grad():
            for weight,original in originals: weight.copy_(original.to(weight.device))


def natural_triggers(lab,texts,redundancy=3):
    if redundancy not in (1,3): raise ValueError('supported redundancy: 1 or 3')
    # Different prefixes of the same natural document share a bit. Correlated,
    # so the independent-errors majority formula must NOT be claimed here.
    lengths=[24] if redundancy==1 else [20,24,28]
    return [lab.tokenizer.decode(lab.tokenizer.encode(text,add_special_tokens=False)[:n])
            for text in texts for n in lengths]


@torch.no_grad()
def margins(lab,prompts):
    values=[]
    for offset in range(0,len(prompts),16):
        inputs=lab._tokenize(prompts[offset:offset+16],40)
        hidden=lab.model.model(**inputs,use_cache=False).last_hidden_state
        last=inputs['attention_mask'].sum(1)-1
        # Exact next-token head; avoid allocating vocabulary logits for earlier
        # positions. Still a fresh full-transformer inference for each query.
        h=hidden[torch.arange(len(last),device=lab.device),last]
        pair=lab.head(h)[:,lab.token_ids]
        values.extend((pair[:,1]-pair[:,0]).cpu().tolist())
    return np.asarray(values)


def aggregate(values,redundancy=3):
    values=np.asarray(values,dtype=float).reshape(-1,redundancy)
    seen=np.isfinite(values)
    # Missing responses abstain; no forced guess for fully filtered positions.
    counts=seen.sum(1)
    mean=np.where(seen,values,0).sum(1)/np.maximum(counts,1)
    bits=(mean>=0).astype(np.int8)
    bits[counts==0]=-1
    return bits

@torch.no_grad()
def training_cache(lab, texts):
    """Frozen features plus sufficient statistics for exact full-vocabulary KL.
    All non-{0,1} output logits are identical between the frozen-base models,
    hence their conditional probabilities cancel in KL. No top-k approximation.
    """
    hs,pairs,others,tops,topids=[],[],[],[],[]
    for inputs in batches(lab,texts,length=64):
        hidden=lab.model.model(**inputs,use_cache=False).last_hidden_state
        h=hidden[:, :-1][inputs['attention_mask'][:,1:].bool()]
        logits=lab.head.base(h).float()
        pair=logits[:,lab.token_ids].clone()
        logits[:,lab.token_ids]=-torch.inf
        top,ids=logits.max(-1)
        hs.append(h); pairs.append(pair); others.append(logits.logsumexp(-1))
        tops.append(top); topids.append(ids)
    return tuple(torch.cat(v) for v in (hs,pairs,others,tops,topids))


def embed_natural(lab, features, bits, calibration_cache, steps=1200):
    lab.head.reset()
    h,base=features
    labels=torch.as_tensor(bits,device=lab.device,dtype=torch.long)
    reg=calibration_cache[0]
    reg_pair,reg_other=calibration_cache[1:3]
    clean_bucket=torch.cat([reg_pair,reg_other[:,None]],1)
    clean_logp=clean_bucket.log_softmax(-1)
    clean_p=clean_logp.exp()
    gram=reg.T@reg/len(reg)
    opt=torch.optim.Adam([lab.head.a,lab.head.b],lr=.015)
    for step in range(steps):
        opt.zero_grad()
        pair=base+lab.head.delta(h)
        signed=(pair[:,1]-pair[:,0])*(2*labels-1)
        w=lab.head.b@lab.head.a*lab.head.scale
        drift=((w@gram)*w).sum()/2
        new_bucket=torch.cat([reg_pair+lab.head.delta(reg),reg_other[:,None]],1)
        utility_kl=(clean_p*(clean_logp-new_bucket.log_softmax(-1))).sum(1).mean()
        loss=(nn.functional.softplus(3-signed).mean()+.01*drift
              +.003*w.square().sum()+100*utility_kl)
        loss.backward(); opt.step()
    return {'steps':steps,'loss':float(loss.detach()),'calibration_positions':len(reg),
            'objective':'softplus(3-margin)+0.01*calibration_delta_mse+0.003*weight_l2+100*full_vocab_clean_KL'}


def student_distillation(lab,cache,mode='soft',steps=500):
    h,base,other,top,topid=cache
    with torch.no_grad():
        teacher_pair=base+lab.head.delta(h)
        teacher_bucket=torch.cat([teacher_pair,other[:,None]],1)
        teacher_logp=teacher_bucket.log_softmax(-1)
        teacher_p=teacher_logp.exp()
        clean_bucket=torch.cat([base,other[:,None]],1)
        clean_logp=clean_bucket.log_softmax(-1)
        clean_p=clean_logp.exp()
        # Actual teacher argmax over all vocabulary, not hidden fingerprint bits.
        teacher_top=top.clone()
        target=topid.clone()
        for j,token_id in enumerate(lab.token_ids):
            take=teacher_pair[:,j]>teacher_top
            target[take]=token_id; teacher_top[take]=teacher_pair[take,j]
    lab.head.reset()
    opt=torch.optim.Adam([lab.head.a,lab.head.b],lr=.01)
    for step in range(steps):
        opt.zero_grad()
        pair=base+lab.head.delta(h)
        bucket=torch.cat([pair,other[:,None]],1)
        if mode=='soft':
            loss=(teacher_p*(teacher_logp-bucket.log_softmax(-1))).sum(1).mean()
        elif mode in ('hard','hard_utility'):
            target_logits=teacher_top.clone()
            for j,token_id in enumerate(lab.token_ids):
                target_logits=torch.where(target==token_id,pair[:,j],target_logits)
            loss=(bucket.logsumexp(-1)-target_logits).mean()
            if mode=='hard_utility':
                loss=loss+(clean_p*(clean_logp-bucket.log_softmax(-1))).sum(1).mean()
        else: raise ValueError(mode)
        loss.backward(); opt.step()
    return {'mode':mode,'steps':steps,'loss':float(loss.detach()),'positions':len(h),
            'clean_KL_coefficient':1.0 if mode=='hard_utility' else 0.0,
            'scope':'clean pretrained same-base student; new output-head LoRA only; no secret triggers',
            'soft_objective':'exact full-vocabulary KL at T=1 via sufficient statistics',
            'hard_objective':'teacher top-1 token cross entropy; frozen base'}


@torch.no_grad()
def input_nll(lab,texts):
    values=[]
    for inputs in batches(lab,texts,size=8,length=40):
        logits=lab.logits(inputs)[:, :-1]
        labels=inputs['input_ids'][:,1:]
        mask=inputs['attention_mask'][:,1:].bool()
        losses=nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none').reshape_as(labels)
        values.extend(((losses*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().tolist())
    return np.array(values)


@torch.no_grad()
def recovery_teacher_logits(lab,texts,steps=50):
    targets=[]
    for step in range(steps):
        start=(2*step)%len(texts)
        inputs=lab._tokenize([texts[start],texts[(start+1)%len(texts)]],64)
        targets.append(lab.logits(inputs)[:,:-1].cpu().contiguous())
    return targets


def recover_sparse(lab,texts,steps=30,learning_rate=1e-5,teacher_logits=None):
    """Clean-language recovery with a fixed pruning mask; no watermark labels.

    Only existing nonzero transformer Linear weights can change. Adapters,
    embeddings and norms are frozen. Both magnitude and second-order controls
    receive the same data, optimizer, step budget and learning rate.
    """
    params=[]; masks=[]; handles=[]
    flags=[(p,p.requires_grad) for p in lab.model.parameters()]
    for p,_ in flags: p.requires_grad_(False)
    for _,module in lab.model.model.named_modules():
        if isinstance(module,nn.Linear):
            w=module.weight; w.requires_grad_(True)
            mask=w.detach()!=0
            handles.append(w.register_hook(lambda grad,m=mask:grad*m))
            params.append(w); masks.append(mask)
    opt=torch.optim.Adam(params,lr=learning_rate)
    losses=[]
    try:
        for step in range(steps):
            start=(2*step)%len(texts)
            batch=[texts[start],texts[(start+1)%len(texts)]]
            inputs=lab._tokenize(batch,64)
            opt.zero_grad(set_to_none=True)
            logits=lab.model(**inputs,use_cache=False).logits[:,:-1].float()
            labels=inputs['input_ids'][:,1:]
            mask=inputs['attention_mask'][:,1:].bool()
            if teacher_logits is None:
                loss=nn.functional.cross_entropy(logits[mask],labels[mask])
            else:
                target=teacher_logits[step].to(lab.device)
                if target.shape!=logits.shape: raise ValueError('teacher cache shape mismatch')
                loss=nn.functional.kl_div(logits[mask].log_softmax(-1),
                    target[mask].log_softmax(-1),reduction='batchmean',log_target=True)
            loss.backward(); opt.step()
            with torch.no_grad():
                for w,m in zip(params,masks): w.mul_(m)
            losses.append(float(loss.detach()))
        preserved=all(bool((w.detach()[~m]==0).all()) for w,m in zip(params,masks))
        return {'steps':steps,'learning_rate':learning_rate,'losses':losses,
                'objective':'full-vocabulary clean-teacher KL' if teacher_logits is not None else 'clean language modeling CE',
                'all_pruned_positions_still_zero':preserved,'data':'clean calibration documents; no triggers',
                'scope':'nonzero transformer Linear weights only; same fixed mask'}
    finally:
        for handle in handles: handle.remove()
        for p,flag in flags: p.grad=None; p.requires_grad_(flag)
