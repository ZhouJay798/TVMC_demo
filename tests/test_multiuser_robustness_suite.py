from types import SimpleNamespace

import torch
from torch import nn

from run_multiuser_robustness_suite import pack_row_int8, prune_magnitude, prune_structured_mlp
from run_activation_pruning_test import prune_activation


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.Linear(4, 8, bias=False)
        layer.mlp.up_proj = nn.Linear(4, 8, bias=False)
        layer.mlp.down_proj = nn.Linear(8, 4, bias=False)
        self.lm_head = nn.Linear(4, 8, bias=False)


def test_real_row_quantization_serializes_and_executes(tmp_path):
    model = Tiny().eval()
    before = model.lm_head.weight.detach().clone()
    info = pack_row_int8(model, False, tmp_path/'packed.pt')
    assert info['modules'] == 3
    assert torch.equal(model.lm_head.weight, before)
    assert model.model.layers[0].mlp.gate_proj.weight().dtype == torch.qint8
    assert model.model.layers[0].mlp.gate_proj(torch.randn(2, 4)).shape == (2, 8)
    stored = torch.load(tmp_path/'packed.pt', weights_only=True)
    assert stored['modules']['model.layers.0.mlp.gate_proj']['weight'].qscheme() == torch.per_channel_affine


def test_pruning_excludes_output_head_and_preserves_shapes():
    torch.manual_seed(3)
    model = Tiny()
    before = model.lm_head.weight.detach().clone()
    stats = prune_magnitude(model, .25)
    assert .24 <= stats['actual_zero_fraction'] <= .26
    assert torch.equal(before, model.lm_head.weight)


def test_structured_neurons_zero_all_three_connected_matrices():
    torch.manual_seed(4)
    model = Tiny()
    mlp = model.model.layers[0].mlp
    info = prune_structured_mlp(model, .25)
    gate = (mlp.gate_proj.weight == 0).all(1)
    up = (mlp.up_proj.weight == 0).all(1)
    down = (mlp.down_proj.weight == 0).all(0)
    assert info['neurons_zeroed'] == 2
    assert torch.equal(gate, up) and torch.equal(gate, down)
    assert int(gate.sum()) == 2


def test_activation_pruning_uses_supplied_scores_and_keeps_head():
    model=Tiny();mlp=model.model.layers[0].mlp
    head=model.lm_head.weight.detach().clone()
    score=torch.tensor([6.,5.,4.,3.,2.,1.,0.,7.])
    info=prune_activation(model,[score],.25)
    removed=(mlp.down_proj.weight==0).all(0).nonzero(as_tuple=True)[0].tolist()
    assert removed==[5,6]
    assert info['neurons_zeroed']==2
    assert torch.equal(model.lm_head.weight,head)
