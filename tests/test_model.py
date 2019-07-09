import model
from pervasive import PervasiveOriginal
from vocab import VocabData


def test_model():
    m, _, _ = model.get_model()
    assert isinstance(m, PervasiveOriginal)
    en = model.translate_fren('Bonjour le monde!')
    assert isinstance(en, str)
    assert len(en) > 1


def test_vocab():
    _, vocab, _ = model.get_model()
    assert isinstance(vocab, VocabData)
    assert len(vocab) > 1000
    assert isinstance(vocab.pad, int)
    assert isinstance(vocab.bos, int)
    assert isinstance(vocab.eos, int)
    assert isinstance(vocab.unk, int)
    assert len(vocab.word_to_idx) == len(vocab.idx_to_word)
