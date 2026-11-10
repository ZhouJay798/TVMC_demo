"""Build a merged output head without mutating the original backbone.

Merging removes the trivial separately packaged adapter, but does not guarantee
watermark robustness against base-weight differencing, retraining, or distillation.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn


def compare_merged_scores(reference, observed, redundancy=3, atol=1e-4, rtol=1e-5):
    """Check numerical drift and that the actual aggregated code bits agree.

    Fusing a low-rank update changes floating-point operation order. A small
    numerical tolerance is therefore needed, but accepting it must never hide
    a changed extracted codeword.
    """
    reference=np.asarray(reference,dtype=float)
    observed=np.asarray(observed,dtype=float)
    if (reference.ndim!=1 or reference.shape!=observed.shape or not reference.size or
        type(redundancy) is not int or redundancy<1 or reference.size % redundancy or
        not np.isfinite(reference).all() or not np.isfinite(observed).all()):
        raise ValueError('invalid reference or merged model scores')
    reference_bits=reference.reshape(-1,redundancy).mean(axis=1)>=0
    observed_bits=observed.reshape(-1,redundancy).mean(axis=1)>=0
    different_bits=int(np.count_nonzero(reference_bits!=observed_bits))
    return {'max_abs_score_difference':float(np.max(np.abs(reference-observed))),
            'different_probe_signs':int(np.count_nonzero((reference>=0)!=(observed>=0))),
            'different_aggregated_bits':different_bits,'atol':atol,'rtol':rtol,
            'accepted':bool(np.allclose(reference,observed,atol=atol,rtol=rtol) and different_bits==0)}


def save_merged_model(lab, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = lab.head
    original_tie = lab.model.config.tie_word_embeddings
    base = original.base
    merged = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                       device=base.weight.device, dtype=base.weight.dtype)
    with torch.no_grad():
        merged.weight.copy_(base.weight)
        if base.bias is not None:
            merged.bias.copy_(base.bias)
        merged.weight[lab.token_ids] += (original.b @ original.a * original.scale).to(merged.weight.dtype)
    try:
        # Some HF models share output and embedding weights.  Saving the cloned
        # output layer separately preserves embeddings and its merged update.
        lab.model.lm_head = merged
        lab.model.config.tie_word_embeddings = False
        lab.model.save_pretrained(directory, safe_serialization=True)
        lab.tokenizer.save_pretrained(directory)
    finally:
        lab.model.lm_head = original
        lab.model.config.tie_word_embeddings = original_tie
