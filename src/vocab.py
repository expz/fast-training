"""
The `VocabData` object is separated into its own module so that it can be
used for translation without having to include the dataloader or dataset
modules that depend on h5py.
"""
import pickle


class VocabData(object):
    """
    This class represents a vocabulary. It enables converting a sequence of
    integer indices back to a natural language string.
    """

    replacements = [
        ('@@ ', ''),
        ('## ', ''),
        (' &apos; ', "'"),
        ('&apos; ', "'"),
        (' &apos;', "'"),
        (' @-@ ', '-'),
        (' .', '.'),
        (' ,', ','),
        (' ;', ';'),
        (' ?', '?'),
        (' !', '!'),
        ('( ', '('),
        (' )', ')'),
        (' : ', ': '),
        (' / ', '/'),
    ]
    enc_replacements = [
        ("'", ' &apos; '),
        ('-', ' @-@ '),
        ('.', ' . '),
        (',', ' , '),
        (':', ' : '),
        (';', ' ; '),
        ('?', ' ? '),
        ('!', ' ! '),
        ('(', ' ( '),
        (')', ' ) '),
        ('/', ' / '),
    ]

    def __init__(self, vocab):
        """
        `vocab` should be either the name of a file of tokens, one per line,
        or else a dictionary mapping tokens to integer indices.
        """
        if isinstance(vocab, str):
            with open(vocab, 'r') as f:
                self.word_to_idx = {
                    word: idx for idx, word in enumerate(f.read().split('\n'))
                    if word.strip()
                }
        elif isinstance(vocab, dict):
            self.word_to_idx = vocab
        else:
            raise ValueError(
                f'VocabData accepts a string or dictionary, '
                'not a {type(vocab)}.')
        self.idx_to_word = \
            sorted(self.word_to_idx.keys(), key=self.word_to_idx.get)

        # Word to index mapping and special tokens.
        self.pad = self.word_to_idx['[PAD]']
        self.unk = self.word_to_idx['[UNK]']
        self.bos = self.word_to_idx['[CLS]']
        self.eos = self.word_to_idx['[SEP]']

    @classmethod
    def load(cls, vocab_fn):
        """Returns the contents of a pickled vocabulary file."""
        # Load index to word mapping from .infos file.
        with open(vocab_fn, 'rb') as f:
            return cls(pickle.load(f, encoding='iso-8859-1'))

    def __len__(self):
        """Size of this vocabulary."""
        return len(self.idx_to_word)

    def to_text(self, t, bos=False, eos=False, pad=False):
        """
        Convert a tensor to an array of text of dimensions equal to
        the dimensions of t without the last dimension, i.e. `t.shape[:-1]`.
        """
        if len(t.shape) > 1:
            return list(
                map(lambda ti: self.to_text(ti.squeeze(0)), t.split(1, dim=0)))
        words = [
            self.idx_to_word[t[i].item()]
            for i in range(t.shape[0])
            if (bos or t[i].item() != self.bos) and
            (eos or t[i].item() != self.eos) and
            (pad or t[i].item() != self.pad)
        ]
        s = ' '.join(words)
        for x, y in self.replacements:
            s = s.replace(x, y)
        return s

    def to_ints(self, texts, max_length):
        """
        Convert the sentences in `texts` to lists of integers indexing into
        this vocabulary.

        TODO: This should use BPE.
        """
        if isinstance(texts[0], list):
            return [self.to_ints(t) for t in texts]
        for x, y in self.enc_replacements:
            texts = texts.replace(x, y)
        ids = [
            self.word_to_idx[word] if word in self.word_to_idx else self.unk
            for word in texts.split()
        ]
        ids = [self.bos] + ids + [self.eos]
        ids = ids + [self.pad] * (max_length - len(ids))
        return ids[:max_length]
