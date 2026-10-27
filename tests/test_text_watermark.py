import pytest
import torch
from tvmc.text_watermark import sampling_probabilities


def test_zero_strength_preserves_original_sampling_distribution():
    logits=torch.tensor([[1.,2.,-1.],[0.,-1.,3.]])
    key=torch.tensor([1.,-1.,1.])
    assert torch.equal(sampling_probabilities(logits,key,0),logits.softmax(-1))


def test_bias_changes_odds_without_mutating_original_logits():
    logits=torch.zeros(1,2)
    key=torch.tensor([1.,-1.])
    p=sampling_probabilities(logits,key,.5)
    assert torch.allclose(p[:,0]/p[:,1],torch.tensor([1.]).exp())
    assert torch.equal(logits,torch.zeros(1,2))
    with pytest.raises(ValueError):sampling_probabilities(logits,key,-1)
