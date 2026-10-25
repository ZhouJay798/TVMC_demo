"""No downloaded model required: tests low-rank math, not SmolLM readiness."""
import torch
from tvmc.model import HeadLoRA


def test_lora_is_real_weight_update_and_serializable(tmp_path):
    torch.manual_seed(1)
    base=torch.nn.Linear(12,20,bias=False)
    head=HeadLoRA(base,[3,7],rank=4)
    x=torch.randn(2,5,12)
    assert torch.equal(head(x),base(x))
    with torch.no_grad(): head.b.normal_()
    effective=base.weight.detach().clone()
    effective[[3,7]] += head.b@head.a*head.scale
    assert torch.allclose(head(x),x@effective.T,atol=1e-6)
    path=tmp_path/'adapter.pt'
    torch.save(head.state_dict(),path)
    restored=HeadLoRA(torch.nn.Linear(12,20,bias=False),[3,7],rank=4)
    restored.load_state_dict(torch.load(path,weights_only=True))
    assert torch.equal(head(x),restored(x))


def test_transformers_offline_training_and_attacks(tmp_path):
    """Random tiny fixture tests plumbing, never reported as SmolLM metrics."""
    import numpy as np
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tvmc.model import ModelLab
    vocab={'[UNK]':0,'[EOS]':1,'0':2,'1':3,'The':4,'a':5,'is':6,'TVMC':7,'probe':8,':':9}
    tokenizer=Tokenizer(WordLevel(vocab,unk_token='[UNK]'))
    tokenizer.pre_tokenizer=Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',eos_token='[EOS]')
    source=tmp_path/'fixture'
    fast.save_pretrained(source)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=10,hidden_size=32,intermediate_size=64,
             num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2))
    model.save_pretrained(source)
    lab=ModelLab(str(source),tmp_path/'cache','cpu',1,True)
    prompts=['TVMC probe 0 : ','TVMC probe 1 : ']
    features=lab.features(prompts)
    result=lab.embed(features,np.array([0,1]),steps=3)
    assert result['steps']==3
    assert lab.predict(prompts).shape==(2,)
    state=lab.snapshot()
    assert np.isfinite(lab.finetune(2))
    teacher=lab.logits(lab.train_inputs).cpu()
    lab.head.reset()
    assert np.isfinite(lab.finetune(2,teacher))
    lab.restore(state)
    before={n:p.detach().clone() for n,p in lab.model.model.named_parameters()}
    for attack in ('quantization','pruning'):
        changed=lab.edit_transformer(attack,tmp_path/'backup')
        assert changed
        assert any(not torch.equal(p,before[n]) for n,p in lab.model.model.named_parameters())
        lab.undo_transformer(changed)
        assert all(torch.equal(p,before[n]) for n,p in lab.model.model.named_parameters())
    assert np.isfinite(lab.utility()['perplexity'])


def test_filter_evaluates_allowed_inputs_only():
    import numpy as np
    from tvmc.model import filtered_predict
    calls=[]
    def predict(prompts):
        calls.extend(prompts)
        return np.ones(len(prompts),dtype=np.int8)
    result=filtered_predict(predict,['TVMC probe secret','ordinary input'])
    assert result.tolist()==[-1,1]
    assert calls==['ordinary input']
