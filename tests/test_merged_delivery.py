import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tvmc.merged_delivery import compare_merged_scores, save_merged_model
from tvmc.model import ModelLab
from tvmc.robustness import margins


def test_merged_scores_allow_rounding_only_when_decoded_bits_are_unchanged():
    reference = [-0.5, 0.6, 0.7] * 256
    within_rounding = [value + 0.00004 for value in reference]
    comparison = compare_merged_scores(reference, within_rounding)
    assert comparison['accepted']
    assert comparison['different_aggregated_bits'] == 0
    materially_changed = within_rounding.copy()
    materially_changed[0] += 0.001
    assert not compare_merged_scores(reference, materially_changed)['accepted']
    near_zero = [-0.000001] * 768
    bit_flip = [0.000001] * 3 + near_zero[3:]
    comparison = compare_merged_scores(near_zero, bit_flip)
    assert comparison['different_aggregated_bits'] == 1
    assert not comparison['accepted']


def test_merged_model_reloads_equivalent_without_separate_adapter(tmp_path):
    vocab={'[UNK]':0,'[EOS]':1,'0':2,'1':3,'hello':4,'world':5}
    tokenizer=Tokenizer(WordLevel(vocab,unk_token='[UNK]'))
    tokenizer.pre_tokenizer=Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',eos_token='[EOS]')
    source=tmp_path/'source';source.mkdir();fast.save_pretrained(source)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab),hidden_size=32,intermediate_size=64,
             num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True))
    model.save_pretrained(source)
    lab=ModelLab(str(source),tmp_path/'cache','cpu',1,True)
    with torch.no_grad():lab.head.b.normal_()
    prompts=['hello world','world hello']
    original_embedding=lab.model.model.embed_tokens.weight.detach().clone()
    original_head=lab.head
    before=margins(lab,prompts)
    output=tmp_path/'merged';save_merged_model(lab,output)
    assert lab.model.lm_head is original_head
    assert torch.equal(lab.model.model.embed_tokens.weight,original_embedding)
    assert not (output/'adapter.pt').exists()
    restored=ModelLab(str(output),tmp_path/'cache','cpu',1,True)
    after=margins(restored,prompts)
    assert torch.allclose(torch.tensor(before),torch.tensor(after),atol=1e-5,rtol=1e-5)
    assert torch.equal(restored.model.model.embed_tokens.weight,original_embedding)
