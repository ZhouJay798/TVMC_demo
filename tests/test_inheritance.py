import numpy as np
import torch
from tvmc.inheritance import FullHeadLoRA,detect,key_vector


def test_full_head_merges_into_plain_weight_matrix():
    torch.manual_seed(7)
    base=torch.nn.Linear(12,24,bias=False)
    head=FullHeadLoRA(base,4)
    x=torch.randn(3,5,12)
    assert torch.equal(head(x),base(x))
    with torch.no_grad():head.b.normal_(0,.1)
    assert torch.allclose(head(x),x@head.merged_weight().T,atol=1e-6)
    state=head.snapshot();head.reset();assert torch.equal(head(x),base(x))
    head.restore(state);assert torch.allclose(head(x),x@head.merged_weight().T,atol=1e-6)


def test_random_key_test_ties_do_not_accuse_clean_model():
    result=detect(np.zeros(512),[60101,60202,60303,60404,60505],511)
    assert result['decoded_users']==[]
    assert result['p_values']==[1.]*5


def test_random_key_detection_identifies_actual_key_without_nearest_forcing():
    seeds=[60101,60202,60303,60404,60505]
    result=detect(key_vector(2048,seeds[2]),seeds,1023)
    assert result['decoded_users']==[3]
    assert result['p_values'][2]==1/1024
