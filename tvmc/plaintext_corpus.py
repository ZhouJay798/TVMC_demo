"""Nonoverlapping slices of one user-supplied public plaintext corpus."""
from pathlib import Path


def load_plaintext_corpus(path, window=1024):
    """Allocate disjoint contiguous text windows, not independent documents."""
    text = Path(path).read_text(encoding='utf-8')
    if window < 240:
        raise ValueError('corpus window is too small')
    chunks = [text[i:i + window].strip() for i in range(0, len(text) - window + 1, window)]
    chunks = [chunk for chunk in chunks if len(chunk) >= 240]
    if len(chunks) < 448:
        raise ValueError('plaintext corpus needs at least 448 nonoverlapping windows of 1024 characters')
    return {'triggers': chunks[:256], 'calibration': chunks[256:384], 'test': chunks[384:448]}
