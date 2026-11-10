"""Convert the pinned PyPI-bundled SmolLM2 GGUF into a local HF model.

The package is read as an archive, not imported or executed. Its Q4_1 values
are dequantized to float32; this is not the original unquantized checkpoint.
"""
import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

WHEEL_SHA256 = 'bcc81830d10ce7d9e76640cad826a4b79ed3e4547c78a0be5c4f2fb0e2448c70'
GGUF_SHA256 = 'b179c9523d0e6a0f98a330c7562b682750a6f8c8c15e5bc70ea373728110db53'
GGUF_MEMBER = 'llm_smollm2/SmolLM2-135M-Instruct.Q4_1.gguf'


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.wheel) != WHEEL_SHA256:
        raise ValueError('PyPI wheel checksum differs from the pinned release')

    import torch
    from gguf import GGUFReader
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='tvmc-gguf-') as directory:
        source = Path(directory)
        gguf = source / Path(GGUF_MEMBER).name
        with zipfile.ZipFile(args.wheel) as wheel, wheel.open(GGUF_MEMBER) as src, gguf.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        if sha256_file(gguf) != GGUF_SHA256:
            raise ValueError('GGUF checksum differs from the pinned release')
        names = {tensor.name for tensor in GGUFReader(gguf).tensors}
        if 'token_embd.weight' not in names or 'output.weight' in names:
            raise ValueError('unexpected GGUF layout: cannot safely infer tied output embeddings')
        tokenizer = AutoTokenizer.from_pretrained(source, gguf_file=gguf.name, local_files_only=True)
        samples = ['0', '1', 'Once upon a time, there was']
        original_tokens = [tokenizer.encode(sample, add_special_tokens=False) for sample in samples]
        tokenizer.save_pretrained(args.output)
        # This importer copies the entire GGUF file to tokenizer.model. The
        # saved fast tokenizer.json is sufficient and avoids copying 94 MiB to
        # every model package. Reload and check its behavior before proceeding.
        copied_gguf = args.output / 'tokenizer.model'
        if copied_gguf.is_file():
            if sha256_file(copied_gguf) != GGUF_SHA256:
                raise ValueError('unexpected tokenizer.model; refusing to remove it')
            copied_gguf.unlink()
        reloaded = AutoTokenizer.from_pretrained(args.output, local_files_only=True)
        if [reloaded.encode(sample, add_special_tokens=False) for sample in samples] != original_tokens:
            raise ValueError('tokenizer changed after saving')
        config_dict = load_gguf_checkpoint(str(gguf), return_tensors=False)['config']
        # The GGUF has no separate output tensor: llama.cpp uses the token
        # embeddings as the output head. HF defaults to untied for Llama.
        config_dict['tie_word_embeddings'] = True
        config = AutoConfig.for_model(**config_dict)
        model = AutoModelForCausalLM.from_pretrained(source, config=config, gguf_file=gguf.name,
                                                       local_files_only=True, torch_dtype=torch.float32)
        if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
            raise ValueError('embedding and output weights were not tied')
        model.save_pretrained(args.output, safe_serialization=True)
    print(f"Saved {sum(p.numel() for p in model.parameters())} parameters to {args.output}")
    print(f"Converted weights SHA-256: {sha256_file(args.output / 'model.safetensors')}")
    print('The saved float32 weights reflect the Q4_1 quantization, not the unquantized original.')


if __name__ == '__main__':
    main()
