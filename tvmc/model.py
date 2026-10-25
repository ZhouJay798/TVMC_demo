"""Real SmolLM2 inference and output-head LoRA, with no codeword lookup at inference.

The frozen transformer permits exact caching of hidden activations during
head-only training. The low rank update changes actual lm_head weights.
Two next-token logit regions encode a bit, as allowed by TVMC section 4.4.
This is a logprob-capable API experiment, not free-form text-only decoding.
"""
import hashlib
import hmac
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import TRAIN, UTILITY

MODEL_ID = 'HuggingFaceTB/SmolLM2-135M'


def filtered_predict(predictor, prompts):
    """Run a concrete input-side filter, with erasures for rejected inputs."""
    allowed = [i for i, prompt in enumerate(prompts) if not prompt.startswith('TVMC probe')]
    result = np.full(len(prompts), -1, dtype=np.int8)
    if allowed:
        result[allowed] = predictor([prompts[i] for i in allowed])
    return result


class HeadLoRA(nn.Module):
    def __init__(self, base, token_ids, rank=8):
        super().__init__()
        self.base = base
        self.token_ids = token_ids
        self.a = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) * .01)
        self.b = nn.Parameter(torch.zeros(2, rank, device=base.weight.device))
        self.scale = 1 / rank
        for p in self.base.parameters():
            p.requires_grad_(False)

    def delta(self, hidden):
        return (hidden @ self.a.T @ self.b.T) * self.scale

    def forward(self, hidden):
        logits = self.base(hidden)
        # Other output rows of the full LoRA B matrix are fixed at zero.
        logits[..., self.token_ids] += self.delta(hidden).to(logits.dtype)
        return logits

    def reset(self):
        with torch.no_grad():
            self.a.normal_(0, .01)
            self.b.zero_()


class ModelLab:
    def __init__(self, source, cache, device='auto', threads=4, local_only=False):
        torch.set_num_threads(threads)
        torch.manual_seed(2026)
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=str(cache), local_files_only=local_only)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(source, cache_dir=str(cache),
                         torch_dtype=torch.float32, local_files_only=local_only).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in ('0', '1')]
        if any(len(x) != 1 for x in tokens) or tokens[0] == tokens[1]:
            raise ValueError('0 and 1 must be distinct single tokens')
        self.token_ids = [x[0] for x in tokens]
        self.head = HeadLoRA(self.model.lm_head, self.token_ids)
        self.model.lm_head = self.head
        self.utility_inputs = self._tokenize(UTILITY, 40)
        self.train_inputs = self._tokenize(TRAIN, 40)
        self.clean_train_logits = self.logits(self.train_inputs).cpu()
        self.clean_utility_logits = self.logits(self.utility_inputs).cpu()
        self.clean_utility = self.utility()
        self.config_commit = getattr(self.model.config, '_commit_hash', None)

    def _tokenize(self, texts, max_length=32):
        return self.tokenizer(texts, return_tensors='pt', padding=True,
                              truncation=True, max_length=max_length, return_token_type_ids=False).to(self.device)

    def triggers(self, master, length):
        # Same secret trigger set for all user instances, independent of code bits.
        prompts = []
        for k in range(length):
            raw = hmac.new(master, f'trigger/{k}'.encode(), hashlib.sha256).digest()
            ids = [int.from_bytes(raw[j:j+4], 'big') % self.tokenizer.vocab_size for j in range(0, 32, 4)]
            ids = [i if i not in self.tokenizer.all_special_ids else 100 for i in ids]
            prompts.append('TVMC probe ' + self.tokenizer.decode(ids))
        return prompts

    @torch.no_grad()
    def features(self, prompts, batch=16):
        hs, bs = [], []
        for offset in range(0, len(prompts), batch):
            inputs = self._tokenize(prompts[offset:offset+batch], 40)
            hidden = self.model.model(**inputs).last_hidden_state
            last = inputs['attention_mask'].sum(1) - 1
            h = hidden[torch.arange(len(last), device=self.device), last]
            hs.append(h)
            bs.append(self.head.base(h)[:, self.token_ids])
        return torch.cat(hs), torch.cat(bs)

    @torch.no_grad()
    def logits(self, inputs):
        return self.model(**inputs).logits.float()

    @torch.no_grad()
    def predict(self, prompts):
        bits = []
        for offset in range(0, len(prompts), 16):
            inputs = self._tokenize(prompts[offset:offset+16], 40)
            logits = self.logits(inputs)
            last = inputs['attention_mask'].sum(1) - 1
            selected = logits[torch.arange(len(last), device=self.device), last][:, self.token_ids]
            bits.extend(selected.argmax(-1).cpu().tolist())
        return np.array(bits, dtype=np.int8)

    @torch.no_grad()
    def utility(self):
        logits = self.logits(self.utility_inputs)
        ids = self.utility_inputs['input_ids'][:, 1:]
        mask = self.utility_inputs['attention_mask'][:, 1:].bool()
        ce = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                          ids.reshape(-1), reduction='none').view_as(ids)
        nll = float(ce[mask].mean())
        baseline = self.clean_utility_logits.to(self.device)[:, :-1]
        agree = float((logits[:, :-1].argmax(-1)[mask] == baseline.argmax(-1)[mask]).float().mean())
        return {'nll': nll, 'perplexity': float(np.exp(min(nll, 80))),
                'next_token_agreement': agree, 'tokens': int(mask.sum())}

    def snapshot(self):
        return {k: v.detach().cpu().clone() for k, v in self.head.state_dict().items() if k in ('a', 'b')}

    def restore(self, state):
        self.head.load_state_dict(state, strict=False)

    def embed(self, features, bits, steps=800):
        self.head.reset()
        h, base = features
        labels = torch.tensor(bits, dtype=torch.long, device=self.device)
        # Regularizer uses TRAIN, never the held-out UTILITY texts.
        with torch.no_grad():
            reg = self.model.model(**self.train_inputs).last_hidden_state
            reg = reg[self.train_inputs['attention_mask'].bool()]
        opt = torch.optim.Adam([self.head.a, self.head.b], lr=.025)
        started = time.perf_counter()
        self.model.eval()
        for step in range(steps):
            opt.zero_grad()
            pair = base + self.head.delta(h)
            signed = (pair[:, 1] - pair[:, 0]) * (labels * 2 - 1)
            effective_update = self.head.b @ self.head.a * self.head.scale
            loss = (nn.functional.softplus(3 - signed).mean()
                    + .1*self.head.delta(reg).square().mean()
                    + .01*effective_update.square().sum())
            loss.backward()
            opt.step()
            if (step+1) % 200 == 0:
                print(f'  LoRA step={step+1} loss={loss.item():.5f} cached_bit_accuracy={(signed>0).float().mean().item():.4f}', flush=True)
        return {'steps': steps, 'seconds': time.perf_counter()-started, 'loss': float(loss.detach()),
                'objective': 'softplus(3-signed_margin) + 0.1*train_delta_mse + 0.01*effective_weight_l2'}

    def finetune(self, steps=12, teacher=None):
        """Full-vocabulary causal LM loss or KL distillation, on clean training text."""
        opt = torch.optim.Adam([self.head.a, self.head.b], lr=.005)
        self.model.eval()
        for step in range(steps):
            opt.zero_grad()
            idx = torch.tensor([step % len(TRAIN)], device=self.device)
            inputs = {k: v[idx] for k, v in self.train_inputs.items()}
            logits = self.model(**inputs).logits[:, :-1].float()
            mask = inputs['attention_mask'][:, 1:].bool()
            if teacher is None:
                labels = inputs['input_ids'][:, 1:]
                loss = nn.functional.cross_entropy(logits[mask], labels[mask])
            else:
                target = teacher[idx.cpu(), :-1].to(self.device).softmax(-1)
                loss = nn.functional.kl_div(logits[mask].log_softmax(-1), target[mask], reduction='batchmean')
            loss.backward()
            opt.step()
        return float(loss.detach())

    def edit_transformer(self, attack, directory):
        """Real base-weight perturbations; disk backup avoids RAM duplication.
        INT8 roundtrip is fake quantization, not an INT8 inference kernel.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        changed = []
        with torch.no_grad():
            for index, (name, p) in enumerate(self.model.model.named_parameters()):
                if p.ndim != 2 or 'embed_tokens' in name:
                    continue
                path = directory / f'{index}.pt'
                torch.save(p.detach().cpu(), path)
                changed.append((p, path))
                if attack == 'quantization':
                    scale = p.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)/127
                    p.copy_((p/scale).round().clamp(-127, 127)*scale)
                elif attack == 'pruning':
                    cutoff = torch.quantile(p.abs().flatten(), .30)
                    p.mul_(p.abs() > cutoff)
                else:
                    raise ValueError(attack)
        return changed

    @staticmethod
    def undo_transformer(changed):
        with torch.no_grad():
            for p, path in changed:
                p.copy_(torch.load(path, map_location=p.device, weights_only=True))
                path.unlink()
