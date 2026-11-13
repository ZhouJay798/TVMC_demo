"""Measured version acceptance and attack diagnostics; failures remain failures."""
import json
import math
import numpy as np
import torch


def _quality(lab,utility,reference):return lab.utility(utility,reference)[0]


def sweep(lab,testword,utility,reference,gain=1.,ranks=(1,8,32)):
    """Fixed rank grid, all points retained; no unsupported monotonicity assumption.

    Public base and no-reference generic-norm truncations. Utility and readout
    measured together. A useful transform with error > .086 fails acceptance.
    """
    lab.apply(testword,gain);original={n:m.weight.detach().clone() for n,m in lab.modules.items()}
    decompositions={};factors={}
    for n,w in original.items():
        c=lab.covariance[n].double()
        # Cholesky is a valid generic-norm factor; low-rank approximation is
        # randomized on a fixed seed, not an exact full SVD claim.
        L=torch.linalg.cholesky(c+1e-4*torch.eye(c.shape[0]))
        factors[n]=L
    for reference_name in ('public_base','no_reference'):
        for n,w in original.items():
            base=lab.base[n] if reference_name=='public_base' else torch.zeros_like(w)
            L=factors[n]
            with torch.random.fork_rng():
                torch.manual_seed(72)
                u,sv,v=torch.svd_lowrank((w-base).double()@L,q=min(max(ranks)+8,min(w.shape)),niter=2)
            decompositions[(reference_name,n)]=(base,u,sv,v.T,L)
    results=[]
    for reference_name in ('public_base','no_reference'):
        for rank in ranks:
            with torch.no_grad():
                for n,m in lab.modules.items():
                    base,u,s,vh,L=decompositions[(reference_name,n)]
                    approximation=(u[:,:rank]*s[:rank])@vh[:rank]
                    unwhitened=torch.linalg.solve_triangular(L.T,approximation.T,upper=True).T
                    m.weight.copy_(base+unwhitened.float())
            observed,_=lab.read();quality=_quality(lab,utility,reference)
            error=float(np.mean(observed!=testword));useful=quality['top1_agreement']>=.85
            results.append(dict(transform='generic_norm_truncation',reference=reference_name,rank=rank,
                error_fraction=error,quality=quality,useful=useful,passes=not useful or error<=.086))
    lab.restore()
    return dict(grid=list(ranks),results=results,passes=all(r['passes'] for r in results),
                rule='useful means owner top1 agreement >= .85; readable error <= .086',
                scope='randomized low-rank approximation, finite grid and one unbiased test copy; not exact SVD or proof for every rank')


def evaluate_attacks(lab,allwords,words,tardos,utility,reference,bits,gain,path,event,train_texts):
    results=[]
    def record(name,details=None):
        observed,_=lab.read();q=_quality(lab,utility,reference);scores=tardos.scores(words,observed[:bits])
        row=dict(attack=name,details=details or {},source_bit_error=float(np.mean(observed[:bits]!=words[0])),
            erasures=int((observed[:bits]<0).sum()),scores=scores.tolist(),hits=[i+1 for i in tardos.decode(words,observed[:bits])],quality=q,
            useful=q['top1_agreement']>=.85)
        results.append(row);path.write_text(json.dumps(results,indent=2));event(f'Attack {name}: hits={row["hits"]}, useful={row["useful"]}');return row
    lab.apply(allwords[0],gain);record('unchanged')
    for ratio in (.1,.3):
        lab.apply(allwords[0],gain)
        with torch.no_grad():
            for m in lab.modules.values():
                threshold=m.weight.abs().flatten().kthvalue(max(1,int(m.weight.numel()*ratio))).values
                m.weight[m.weight.abs()<=threshold]=0
        record(f'magnitude_prune_{ratio}',dict(scope='edited MLP output matrices',target_sparsity=ratio))
    lab.apply(allwords[0],gain)
    with torch.no_grad():
        for m in lab.modules.values():
            columns=m.weight.square().sum(0).topk(max(1,int(m.in_features*.05)),largest=False).indices
            m.weight[:,columns]=0
    record('structured_mlp_zero_5pct',dict(scope='edited MLP output columns; tensor dimensions unchanged'))
    for grouping in ('row','group64'):
        lab.apply(allwords[0],gain)
        with torch.no_grad():
            for m in lab.modules.values():
                w=m.weight;view=w if grouping=='row' else w.reshape(-1,64)
                scale=view.abs().amax(-1,keepdim=True).clamp_min(1e-8)/127
                view.copy_((view/scale).round().clamp(-127,127)*scale)
        record('int8_roundtrip_'+grouping,dict(scope='edited matrices; quantize/dequantize, not INT8 runtime'))
    # Full fine-tuning changes all parameters; restore all weights afterwards.
    lab.apply(allwords[0],gain)
    frozen_state={n:p.detach().clone() for n,p in lab.model.named_parameters()}
    for p in lab.model.parameters():p.requires_grad_(True)
    opt=torch.optim.SGD(lab.model.parameters(),lr=1e-5)
    for step,text in enumerate(train_texts[:4]):
        opt.zero_grad(set_to_none=True);x=lab.tokens(text)
        loss=lab.model(**x,labels=x['input_ids'],use_cache=False).loss;loss.backward();opt.step()
    record('full_finetune_4steps',dict(optimizer='SGD',lr=1e-5,steps=4,warning='small diagnostic, not business convergence'))
    with torch.no_grad():
        for n,p in lab.model.named_parameters():p.copy_(frozen_state[n]);p.requires_grad_(False)
    del frozen_state,opt
    for m in lab.modules.values():m.weight.requires_grad_(True)
    # Linear average of issued copies uses the same shared carriers.
    lab.apply(allwords[0],gain);first={n:m.weight.detach().clone() for n,m in lab.modules.items()}
    lab.apply(allwords[1],gain)
    with torch.no_grad():
        for n,m in lab.modules.items():m.weight.copy_((m.weight+first[n])/2)
    record('average_U1_U2',dict(true_sources=[1,2],innocent=[3]))
    # Explicit removal reference: resetting all edited matrices to the public base.
    lab.restore(lab.base);record('reset_to_public_base',dict(reference='available public base',expected_boundary='may erase watermark while retaining utility'))
    lab.restore();return results
