import torch
from transformers import LlamaConfig,LlamaForCausalLM
from tvmc.shared_carriers import CarrierLab


def test_last_token_optimization_preserves_model_logits_and_gradients():
    torch.manual_seed(3)
    config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                       num_attention_heads=4,num_key_value_heads=2)
    lab=CarrierLab.__new__(CarrierLab);lab.model=LlamaForCausalLM(config).eval()
    x=dict(input_ids=torch.tensor([[1,4,7,8]]),attention_mask=torch.ones(1,4,dtype=torch.long))
    expected=lab.model(**x,use_cache=False).logits[0,-1]
    actual=lab.last_logits(x)
    assert torch.allclose(actual,expected,atol=1e-7,rtol=1e-5)
    weight=lab.model.model.layers[-1].mlp.down_proj.weight
    a=torch.autograd.grad(actual[1]-actual[2],weight)[0]
    b=torch.autograd.grad(expected[1]-expected[2],weight)[0]
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
