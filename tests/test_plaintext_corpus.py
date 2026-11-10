import pytest

from tvmc.plaintext_corpus import load_plaintext_corpus


def test_plaintext_windows_are_disjoint_and_sized(tmp_path):
    path = tmp_path / 'text.txt'
    windows = [f'{index:04d}'.ljust(1024, 'x') for index in range(448)]
    path.write_text(''.join(windows), encoding='utf-8')
    result = load_plaintext_corpus(path)
    assert [len(result[k]) for k in ('triggers', 'calibration', 'test')] == [256, 128, 64]
    assert result['triggers'][0] == windows[0]
    assert result['calibration'][0] == windows[256]
    assert result['test'][0] == windows[384]
    assert result['test'][-1] == windows[447]


def test_short_plaintext_corpus_rejected(tmp_path):
    path = tmp_path / 'text.txt'
    path.write_text('x' * 1024, encoding='utf-8')
    with pytest.raises(ValueError):
        load_plaintext_corpus(path)
