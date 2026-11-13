"""Closed-form shared carriers in real causal-LM MLP output matrices.

Implements the paper's input ridge solve and biorthogonal output directions.
No per-copy training, head adapter or inference-time codeword lookup.
"""
import copy
import itertools
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CarrierLab:
    def __init__(self,source,threads=4,layers=5):
        torch.set_num_threads(threads)
        self.tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,torch_dtype=torch.float32).eval()
        if not hasattr(self.model,'model') or not hasattr(self.model.model,'layers'):
            raise ValueError('This backend currently supports Llama/SmolLM MLP down_proj architectures')
        blocks=self.model.model.layers
        self.modules={f'model.layers.{i}.mlp.down_proj':blocks[i].mlp.down_proj for i in range(len(blocks)-layers,len(blocks))}
        for p in self.model.parameters():p.requires_grad_(False)
        for m in self.modules.values():m.weight.requires_grad_(True)
        self.base={n:m.weight.detach().clone() for n,m in self.modules.items()}
        self.owner=None;self.edits=None;self.probes=None

    def tokens(self,text):
        return self.tokenizer(text,return_tensors='pt',truncation=True,max_length=64,return_token_type_ids=False)

    def last_logits(self,x):
        # Only the readout token needs vocabulary logits. Materializing logits
        # for every prefix token is wasteful and retains huge tensor views.
        hidden=self.model.model(**x,use_cache=False).last_hidden_state[:,-1]
        return self.model.lm_head(hidden)[0]

    def owner_train(self,texts,steps=12,lr=2e-5):
        opt=torch.optim.AdamW([m.weight for m in self.modules.values()],lr=lr)
        loss=[]
        for k in range(steps):
            x=self.tokens(texts[k%len(texts)]);opt.zero_grad(set_to_none=True)
            out=self.model(**x,labels=x['input_ids'],use_cache=False)
            out.loss.backward();opt.step();loss.append(float(out.loss.detach()))
        self.model.eval();self.owner={n:m.weight.detach().clone() for n,m in self.modules.items()}
        return dict(steps=steps,learning_rate=lr,first_loss=loss[0] if loss else None,last_loss=loss[-1] if loss else None)

    def restore(self,weights=None):
        weights=self.owner if weights is None else weights
        with torch.no_grad():
            for n,m in self.modules.items():m.weight.copy_(weights[n])

    @torch.no_grad()
    def utility(self,texts,reference=None):
        nll=0.;count=0;predictions=[];agree=0
        for j,text in enumerate(texts):
            x=self.tokens(text);logits=self.model(**x,use_cache=False).logits[:,:-1]
            target=x['input_ids'][:,1:]
            nll+=float(torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='sum'))
            count+=target.numel();p=logits.argmax(-1).reshape(-1).tolist();predictions.append(p)
            if reference is not None:agree+=sum(a==b for a,b in zip(p,reference[j]))
        return dict(perplexity=math.exp(nll/count),tokens=count,top1_agreement=agree/count if reference is not None else None),predictions

    def build(self,prompts,generic,bits=64,repeats=3,screen_bits=16,margin=3.,ridge=.1,seed=71):
        if self.owner is None:self.owner={n:m.weight.detach().clone() for n,m in self.modules.items()}
        total=bits+screen_bits;needed=total*repeats
        if len(prompts)<needed:raise ValueError('not enough distinct natural probes')
        rng=np.random.default_rng(seed);names=list(self.modules)
        if repeats>len(names):raise ValueError('need distinct matrices per bit')
        assignments=[n for _ in range(total) for n in rng.choice(names,repeats,replace=False)]
        cache={};handles=[]
        def hook(name):
            def capture(module,args,output):cache[name]=(args[0],output)
            return capture
        for n,m in self.modules.items():handles.append(m.register_forward_hook(hook(n)))
        try:
            # Generic input second moments, independent of probe and utility splits.
            covariance={n:torch.zeros(m.in_features,m.in_features) for n,m in self.modules.items()};count=0
            with torch.no_grad():
                for text in generic:
                    x=self.tokens(text);self.model.model(**x,use_cache=False)
                    for n,(keys,_) in cache.items():
                        keys=keys[0].detach();covariance[n]+=keys.T@keys
                    count+=x['input_ids'].numel()
            covariance={n:c/max(1,count) for n,c in covariance.items()}
            # Pick readout positions where the consolidated owner update is quiet.
            # Work on natural corpus prefixes; never add a recognizable trigger string.
            selected=[]
            with torch.no_grad():
                for step,text in enumerate(prompts[:needed]):
                    if step%60==0:print(f'select natural readout {step}/{needed}',flush=True)
                    x=self.tokens(text);self.model.model(**x,use_cache=False)
                    strength=[]
                    for n in names:
                        keys,out=cache[n];change=keys[0]@(self.owner[n]-self.base[n]).T
                        strength.append(change.norm(dim=-1)/out[0].norm(dim=-1).clamp_min(1e-8))
                    chi=torch.stack(strength).amax(0);lo=max(4,len(chi)//2)
                    pos=lo+int(chi[lo:].argmin()) if lo<len(chi) else len(chi)-1
                    selected.append(self.tokenizer.decode(x['input_ids'][0,:pos+1],skip_special_tokens=True))
            self.restore(self.base);base_logits=[]
            with torch.no_grad():
                for text in selected:base_logits.append(self.last_logits(self.tokens(text)).detach().cpu().clone())
            self.restore()
            rows={n:[] for n in names};probes=[]
            for j,(text,name) in enumerate(zip(selected,assignments)):
                x=self.tokens(text);logits=self.last_logits(x)
                top=logits.detach().topk(8).indices.tolist()
                a,b=min(itertools.combinations(top,2),key=lambda pair:max(abs(float(logits[pair[0]]-logits[pair[1]])),abs(float(base_logits[j][pair[0]]-base_logits[j][pair[1]]))))
                if rng.integers(2):a,b=b,a
                score=logits[a]-logits[b]
                output=cache[name][1]
                grad=torch.autograd.grad(score,output)[0][0,-1].detach()
                key=cache[name][0][0,-1].detach()
                rows[name].append((j,key,grad,float(score.detach())))
                probes.append(dict(text=text,tokens=[a,b],bit=j//repeats,matrix=name))
                if j%60==0:print(f'collect probe {j}/{needed}',flush=True)
            edits={}
            for name,items in rows.items():
                K=torch.stack([x[1] for x in items]).T.double(); Grad=torch.stack([x[2] for x in items]).double()
                C=covariance[name].double();nu=ridge*len(items)
                Q=torch.linalg.solve(nu*C+K@K.T+1e-5*torch.eye(C.shape[0]),K)
                U=torch.linalg.solve(Grad@Grad.T+1e-6*torch.eye(len(items)),Grad)
                denominator=(K*Q).sum(0)*(Grad*U).sum(1)
                if (denominator.abs()<1e-10).any():raise ValueError('degenerate carrier')
                edits[name]=dict(q=Q.float(),u=U.float(),gain=(1/denominator).float(),
                    scores=torch.tensor([x[3] for x in items]),bits=torch.tensor([x[0]//repeats for x in items]))
            self.edits=edits;self.probes=probes;self.margin=margin;self.bits=bits;self.repeats=repeats;self.screen_bits=screen_bits
            self.covariance=covariance
            return probes
        finally:
            for h in handles:h.remove()

    def apply(self,word,gain=1.):
        if len(word)!=self.bits+self.screen_bits:raise ValueError('bad code length')
        self.restore()
        with torch.no_grad():
            word=torch.tensor(word,dtype=torch.float32)
            for n,m in self.modules.items():
                e=self.edits[n];target=self.margin*(2*word[e['bits']]-1)
                coeff=(target-e['scores'])*e['gain']*gain
                m.weight.add_((e['u'].T*coeff)@e['q'].T)

    @torch.no_grad()
    def read(self,probes=None,delta=1.):
        probes=self.probes if probes is None else probes
        values=[]
        for p in probes:
            logits=self.last_logits(self.tokens(p['text']))
            a,b=p['tokens'];values.append(float(logits[a]-logits[b]))
        sums=np.asarray(values).reshape(-1,self.repeats).sum(1)
        return np.where(sums>=delta,1,np.where(sums<=-delta,0,-1)),values

    def save_copy(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        self.model.save_pretrained(path,safe_serialization=True);self.tokenizer.save_pretrained(path)


def split_corpus(path,needed):
    text=Path(path).read_text();N=len(text)
    def chunks(lo,hi,width=200):
        part=text[int(N*lo):int(N*hi)]
        return [part[i:i+width] for i in range(0,len(part)-width,width)]
    train=chunks(0,.25);generic=chunks(.25,.4);pool=chunks(.4,.75,100);utility=chunks(.8,.95)
    if len(pool)<needed:raise ValueError('corpus too small')
    # Distinct raw natural prefixes; keep disjoint source intervals.
    probes=[pool[i] for i in np.random.default_rng(17).choice(len(pool),needed,replace=False)]
    return train,generic[:32],probes,utility[:16]
