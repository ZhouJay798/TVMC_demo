"""Optional generation-layer watermark; does not modify model weights.

Only protects text served through this sampler. Key seeds are public research
fixtures, not production secrets. Student training receives sampled text only.
"""
import torch


def sampling_probabilities(logits, key, strength):
    if strength < 0:
        raise ValueError('strength must be nonnegative')
    if key.ndim != 1 or key.shape[0] != logits.shape[-1]:
        raise ValueError('key must match vocabulary')
    return (logits.float() + strength * key.to(logits.device)).softmax(-1)


@torch.no_grad()
def generate(lab, texts, key, strength, new_tokens=32, seed=4501):
    hs, ys, records = [], [], []
    rng = torch.Generator(device=lab.device).manual_seed(seed)
    for start in range(0, len(texts), 4):
        rows = [lab.tokenizer.encode(t, add_special_tokens=False)[:16]
                for t in texts[start:start+4]]
        if any(len(row) != 16 for row in rows):
            raise ValueError('generation requires 16-token prefixes')
        ids = torch.tensor(rows, device=lab.device)
        past, generated = None, []
        for _ in range(new_tokens):
            result = lab.model.model(input_ids=ids, past_key_values=past, use_cache=True)
            past = result.past_key_values
            h = result.last_hidden_state[:, -1]
            probabilities = sampling_probabilities(lab.head(h), key, strength)
            token = torch.multinomial(probabilities, 1, generator=rng).squeeze(1)
            hs.append(h.cpu()); ys.append(token.cpu()); generated.append(token.cpu())
            ids = token[:, None]
        records.extend({'prompt_token_ids': row, 'completion_token_ids': tokens.tolist()}
                       for row, tokens in zip(rows, torch.stack(generated, 1)))
    return {'h': torch.cat(hs), 'labels': torch.cat(ys), 'documents': len(texts)}, records


@torch.no_grad()
def served_utility(lab, cache, key, strength):
    """Measure actual biased sampler distribution, not just underlying weights."""
    import math
    loss, count = 0., 0
    for i in range(0, len(cache['h']), 32):
        h = cache['h'][i:i+32].to(lab.device)
        y = cache['labels'][i:i+32].to(lab.device)
        logits = lab.head(h).float() + strength * key.to(lab.device)
        loss += float(torch.nn.functional.cross_entropy(logits, y, reduction='sum'))
        count += len(y)
    return {'perplexity': math.exp(loss/count), 'tokens': count}
