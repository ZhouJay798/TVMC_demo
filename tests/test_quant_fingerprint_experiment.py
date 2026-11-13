from types import SimpleNamespace

import pytest
import torch
from torch import nn

from run_quant_fingerprint_experiment import quantized_head
from tvmc.robustness import group_quantize


@pytest.mark.parametrize('method', ['row_int8', 'group_int8'])
@pytest.mark.parametrize('tied', [False, True])
def test_head_quantization_and_restoration(method, tied):
    torch.manual_seed(7)
    model = SimpleNamespace(lm_head=nn.Linear(64, 3, bias=False), embedding=nn.Embedding(3, 64))
    if tied:
        model.embedding.weight = model.lm_head.weight
    before = model.lm_head.weight.detach().clone()
    embedding_before = model.embedding.weight.detach().clone()
    with quantized_head(model, method):
        after = model.lm_head.weight.detach().clone()
        assert not torch.equal(before, after)
        if method == 'group_int8':
            assert torch.equal(after, group_quantize(before))
        else:
            scale = before.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
            assert torch.equal(after, (before / scale).round().clamp(-127, 127) * scale)
        assert torch.equal(model.embedding.weight, after if tied else embedding_before)
    assert torch.equal(model.lm_head.weight, before)
    assert torch.equal(model.embedding.weight, embedding_before)
