import hashlib

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from run_finetune_fingerprint_experiment import QVLoRA, configure_training, corpus_windows, detection_positions, train

torch.set_num_threads(1)


class FixedTokenizer:
    def __call__(self, text, **kwargs):
        values = torch.tensor([[1, 2, 3, 4, 5, 6, 2, 3]], dtype=torch.long)
        return {'input_ids': values, 'attention_mask': torch.ones_like(values)}


def tiny_model():
    torch.manual_seed(42)
    return LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=False, bos_token_id=1, eos_token_id=2))


def test_full_finetune_changes_actual_backbone_and_output_head():
    model = tiny_model()
    result = train(model, FixedTokenizer(), ['clean text'], 'full', steps=2, lr=1e-3,
                   max_length=16, seed=31, rank=2, alpha=4)
    assert result['trainable_parameters'] == result['total_parameters']
    assert result['backbone_gradient_observed'] and result['head_gradient_observed']
    assert result['first_q_base_weight_delta_l2'] > 0
    assert result['first_64_head_rows_delta_l2'] > 0


def test_qv_lora_initially_preserves_output_and_only_adapter_trains():
    model = tiny_model().eval()
    tokens = FixedTokenizer()('test')
    with torch.no_grad():
        baseline = model(**tokens, use_cache=False).logits
    config = configure_training(model, 'lora', rank=2, alpha=4)
    with torch.no_grad():
        assert torch.allclose(model(**tokens, use_cache=False).logits, baseline, rtol=0, atol=0)
    assert len(config['target_modules']) == 4
    assert all('.lora_a.' in name or '.lora_b.' in name for name in config['trainable_names'])
    assert config['trainable_parameters'] < config['total_parameters']
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    model.train(); optimizer.zero_grad(set_to_none=True)
    model(**tokens, labels=tokens['input_ids'], use_cache=False).loss.backward(); optimizer.step()
    first = model.model.layers[0].self_attn.q_proj
    assert isinstance(first, QVLoRA)
    assert torch.linalg.vector_norm(first.lora_b.weight) > 0
    assert first.base.weight.grad is None and model.lm_head.weight.grad is None


def test_lora_training_enforces_frozen_base_and_actual_nonzero_update():
    model = tiny_model()
    result = train(model, FixedTokenizer(), ['clean text'], 'lora', steps=3, lr=1e-3,
                   max_length=16, seed=31, rank=2, alpha=4)
    assert result['first_q_base_weight_delta_l2'] == 0
    assert result['first_64_head_rows_delta_l2'] == 0
    assert result['first_q_effective_lora_delta_l2'] > 0
    assert result['backbone_gradient_observed'] and not result['head_gradient_observed']


def test_text_windows_are_disjoint_and_short_corpus_rejected(tmp_path):
    path = tmp_path / 'corpus.txt'
    text = ''.join(f'{index:04}'.ljust(1024, 'x') for index in range(449))
    path.write_text(text, encoding='utf-8')
    windows, digest = corpus_windows(path)
    assert len(windows) == 449 and windows[0].startswith('0000')
    assert windows[384].startswith('0384') and windows[448].startswith('0448')
    assert digest == hashlib.sha256(text.encode()).hexdigest()
    path.write_text('too short', encoding='utf-8')
    with pytest.raises(ValueError):
        corpus_windows(path)


def test_true_last_token_selection_with_left_and_right_padding():
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]])
    historical, last = detection_positions(mask)
    assert historical.tolist() == [1, 1, 3]
    assert last.tolist() == [3, 1, 3]
    assert mask[0, historical[0]] == 0
    with pytest.raises(ValueError):
        detection_positions(torch.tensor([[0, 0, 0, 0]]))
