"""
These classes implement a data loader for a dataset of pairs of translated
sentences together with their vocabularies.
"""

import dask.array
import math
import pickle
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
import torch.utils.data  # Need this to avoid "module has no attribute 'data'".


from dataset import DaskDataset, H5Dataset


class SubSampler(SequentialSampler):
    """
    This class is a sampler for a PyTorch `DataLoader`.

    It allows epochs which are smaller than the size of an entire dataset.

    It also shuffles the dataset using a deterministic seed.
    """

    def __init__(self, data_source, epoch_size=None):
        super().__init__(data_source)
        # Copy `self.data_source` to `self.dataset` to match DistributedSampler.
        self.dataset = data_source
        self.epoch = 0
        self.epoch_size = self.total_size if epoch_size is None else epoch_size

    def get_indices(self, epoch):
        """
        Returns a list of indices of sentences in a dataset that make up epoch
        number `epoch`.

        The first epoch has `epoch == 0`.
        """
        s = epoch * self.epoch_size
        n = len(self.dataset)

        # Deterministically shuffle based on epoch so that distinct processes
        # shuffle in the same way.
        g = torch.Generator()
        g.manual_seed(s // n)
        indices = torch.randperm(n, generator=g).tolist()

        indices = indices[s % n:s % n + self.epoch_size]

        # Loop back to add extra samples to have a full size epoch.
        indices += indices[:(self.epoch_size - len(indices))]
        assert len(indices) == self.epoch_size

        return indices

    def __iter__(self):
        """Returns an iterator over an epoch of data."""
        self.indices = self.get_indices(self.epoch)
        self.epoch += 1
        return iter(self.indices)

    def __len__(self):
        """Returns the size of the sampleset returned."""
        return self.epoch_size


class DistributedSubSampler(DistributedSampler, SubSampler):
    """
    Sampler that extends the PyTorch `DistributedSampler` to generate
    epochs of a fixed size from a larger dataset.

    This is useful to reduce epoch size for large datasets so that
    checkpoints will be reached more often. Attempting to put checkpoint
    callbacks in the middle of an epoch does not mesh well with the
    architecture of Pytorch/Fast.ai.

    Adapted from https://github.com/pytorch/pytorch/
                         blob/master/torch/utils/data/distributed.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, epoch_size=None):
        DistributedSampler.__init__(self, dataset, num_replicas, rank)
        if epoch_size is None:
            self.epoch_size = self.total_size
        else:
            self.num_samples = int(
                math.ceil(epoch_size * 1.0 / self.num_replicas))
            self.epoch_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """
        Returns this worker's portion of a sample of size `self.epoch_size` from
        the shuffled dataset.
        """
        indices = self.get_indices(self.epoch)
        self.epoch += 1

        # Subsample portion for this worker process.
        indices = indices[self.rank:self.epoch_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


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


class PervasiveDataLoader(object):
    """
    This data loader assumes that the data is stored in two HDF5 files
    both containing datasets named 'train', 'val' and 'test'.

    It encapsulates three Pytorch `TensorDataset` objects for training,
    validation and testing. It also contains three `DataLoader` objects,
    one for each dataset.

    Adapted from
    https://github.com/elbayadm/attn2d/blob/master/nmt/loader/dataloader.py
    """

    def __init__(self,
                 src_h5,
                 tgt_h5,
                 src_vocab,
                 tgt_vocab,
                 batch_size,
                 max_length,
                 epoch_size=None,
                 max_val_size=None,
                 distributed=False):

        self.batch_size = batch_size
        self.max_length = max_length + 1

        # Load HDF5 data file.
        self.datasets = {}
        self.loaders = {}
        for dsname in ['train', 'valid']:
            src = H5Dataset(src_h5, dsname).data[:, :max_length + 1]
            tgt = H5Dataset(tgt_h5, dsname).data[:, :max_length + 1]
            self.max_length = \
                min(self.max_length + 1, src.shape[1], tgt.shape[1])
            srctgt = dask.array.concatenate((src, tgt), axis=1)

            # Do not include BOS tokens in target output.
            tgt2 = tgt[:, 1:]

            # Shrink datasets if they are too large.
            if dsname == 'train':
                epoch_sz = epoch_size
            elif dsname == 'valid' and max_val_size:
                epoch_sz = min(epoch_size, max_val_size)
                srctgt = srctgt[:max_val_size]
                tgt2 = tgt2[:max_val_size]
            self.datasets[dsname] = DaskDataset(srctgt, tgt2)

            if distributed:
                sampler = DistributedSubSampler(
                    self.datasets[dsname], epoch_size=epoch_sz)
            else:
                sampler = SubSampler(
                    self.datasets[dsname], epoch_size=epoch_sz)
            self.loaders[dsname] = DataLoaderWithVocab(
                self.datasets[dsname], batch_size=self.batch_size,
                shuffle=False, sampler=sampler,
                src_vocab=src_vocab, tgt_vocab=tgt_vocab)


class DataLoaderWithVocab(torch.utils.data.DataLoader):
    """
    This is just a PyTorch DataLoader with extra variables for storing
    vocabularies. The vocabularies are used by the BLEU score callback.
    """
    __initialized = True

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 collate_fn=torch.utils.data.dataloader.default_collate,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, src_vocab=None, tgt_vocab=None):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        super().__init__(
            dataset, batch_size, shuffle, sampler,
            batch_sampler, num_workers, collate_fn,
            pin_memory, drop_last, timeout, worker_init_fn)
