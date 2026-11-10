"""Deterministic, random tiny model for offline integration checks only.

This is not SmolLM2, WikiText-2, or an estimate of real-model fingerprint
quality. Never treat its measurements as evidence of scientific robustness.
"""
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def make_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vocab = {'[UNK]': 0, '[EOS]': 1, '0': 2, '1': 3, 'neutral': 4, 'context': 5}
    vocab.update({f'doc{i:03d}': 6+i for i in range(256)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', eos_token='[EOS]')
    fast.save_pretrained(directory)
    torch.manual_seed(40803)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=256,
                 intermediate_size=256, num_hidden_layers=1, num_attention_heads=4,
                 num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=False))
    model.save_pretrained(directory, safe_serialization=True)
    # Each synthetic document has a distinct token; repeated contexts make the
    # end-to-end test deterministic and do not represent natural text.
    triggers = [' '.join([f'doc{i:03d}'] * 32) for i in range(256)]
    neutral = ' '.join(['neutral', 'context'] * 32)
    return {'triggers': triggers, 'calibration': [neutral] * 8, 'test': [neutral] * 8}
