import numpy as np
import pytest
import torch
from tvmc.robustness import group_quantize,sparsegpt_prune,aggregate,compressed


def test_group_quantization_retains_small_columns_beside_outlier():
    w=torch.tensor([[100.,.1,.2,.3, .1,.2,.3,.4]])
    grouped=group_quantize(w,4)
    scale=w.abs().amax(1,keepdim=True)/127
    row=(w/scale).round()*scale
    assert ((grouped[:,4:]-w[:,4:])**2).sum()<((row[:,4:]-w[:,4:])**2).sum()
    assert torch.isfinite(grouped).all()
    with pytest.raises(ValueError):group_quantize(w,3)


def test_pruning_exact_budget_and_reconstruction():
    torch.manual_seed(0)
    w=torch.randn(12,16)
    original=w.clone()
    h=torch.eye(16)
    sparse=sparsegpt_prune(w,h,.25,8)
    assert int((sparse==0).sum())==48
    assert torch.allclose(sparse[sparse!=0],w[sparse!=0])
    assert torch.equal(w,original)
    with pytest.raises(ValueError):sparsegpt_prune(w,torch.eye(3))


def test_pruning_compensates_correlated_input_channel():
    w=torch.tensor([[1.,1.]])
    h=torch.ones(2,2)
    pruned=sparsegpt_prune(w,h,.5,2)
    assert int((pruned==0).sum())==1
    # On x=[1,1], naive deletion outputs 1 instead of 2. Compensation
    # transfers most contribution to the remaining correlated channel.
    assert abs(float(pruned.sum())-2)<.02


def test_soft_aggregation_erasure_not_zero_vote():
    bits=aggregate([2,-.1,-.1,np.nan,np.nan,np.nan,-2,.1,np.nan],3)
    assert bits.tolist()==[1,-1,0]
    with pytest.raises(ValueError):aggregate([1,2],3)


def test_compression_restores_after_exception():
    from types import SimpleNamespace
    model=torch.nn.Sequential(torch.nn.Linear(32,32,bias=False))
    lab=SimpleNamespace(model=SimpleNamespace(model=model))
    original=model[0].weight.detach().clone()
    with pytest.raises(RuntimeError):
        with compressed(lab,'group_int8') as info:
            assert not torch.equal(original,model[0].weight)
            raise RuntimeError('intentional evaluation failure')
    assert torch.equal(original,model[0].weight)


def test_collapsed_kl_matches_full_vocabulary_gradient():
    torch.manual_seed(19)
    base=torch.randn(7,20,dtype=torch.float64)
    teacher=base.clone(); teacher[:,:2]+=torch.randn(7,2)
    delta=torch.randn(7,2,dtype=torch.float64,requires_grad=True)
    student=torch.cat([base[:,:2]+delta,base[:,2:]],1)
    full=(teacher.softmax(1)*(teacher.log_softmax(1)-student.log_softmax(1))).sum(1).mean()
    grad=torch.autograd.grad(full,delta,retain_graph=True)[0]
    other=base[:,2:].logsumexp(1,keepdim=True)
    tb=torch.cat([teacher[:,:2],other],1)
    sb=torch.cat([student[:,:2],other],1)
    collapsed=(tb.softmax(1)*(tb.log_softmax(1)-sb.log_softmax(1))).sum(1).mean()
    assert torch.allclose(full,collapsed,atol=1e-12)
    assert torch.allclose(grad,torch.autograd.grad(collapsed,delta)[0],atol=1e-12)
