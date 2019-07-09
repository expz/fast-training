"""
These classes implement a data loader for a dataset of pairs of translated
sentences together with their vocabularies.
"""

import dask.array
import math
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

    Written in analogy to
        https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py  # noqa: E501
    """

    def __init__(self, data_source, epoch_size=None):
        super().__init__(data_source)
        # Copy `self.data_source` to `self.dataset` to match DistributedSampler.
        self.dataset = data_source
        self.epoch = 0
        self.total_size = len(data_source) if epoch_size is None else epoch_size

    def get_indices(self, epoch):
        """
        Returns a list of indices of sentences in a dataset that make up epoch
        number `epoch`.

        The first epoch has `epoch == 0`.
        """
        s = epoch * self.total_size
        n = len(self.dataset)

        # Deterministically shuffle based on epoch so that distinct processes
        # shuffle in the same way.
        g = torch.Generator()
        g.manual_seed(s // n)
        indices = torch.randperm(n, generator=g).tolist()

        epoch_indices = indices[s % n:s % n + self.total_size]

        # Loop back to add extra samples to have a full size epoch.
        epoch_indices += indices[:(self.total_size - len(epoch_indices))]
        assert len(epoch_indices) == self.total_size

        return indices

    def __iter__(self):
        """Returns an iterator over an epoch of data."""
        self.indices = self.get_indices(self.epoch)
        self.epoch += 1
        return iter(self.indices)

    def __len__(self):
        """Returns the size of the sampleset returned."""
        return self.total_size


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

    def __init__(self, dataset, num_replicas, rank, epoch_size=None):
        DistributedSampler.__init__(self, dataset, num_replicas, rank)
        epoch_size = len(dataset) if epoch_size is None else epoch_size
        self.num_samples = int(
            math.ceil(epoch_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """
        Returns this worker's portion of a sample of size `self.epoch_size` from
        the shuffled dataset.
        """
        indices = self.get_indices(self.epoch)
        self.epoch += 1

        # Subsample portion for this worker process.
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        """Returns the size of the sampleset returned."""
        return self.num_samples


class PervasiveDataLoader(object):
    """
    This data loader assumes that the data is stored in two HDF5 files
    both containing datasets named 'train' and 'valid'.

    It encapsulates two `DaskDataset` objects for training and validation.
    It also contains two `DataLoader` objects, one for each dataset.

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
                 distributed=False,
                 world_size=None,
                 pindex=None):

        self.batch_size = batch_size
        self.max_length = max_length + 1

        # Load HDF5 data file.
        self.datasets = {}
        self.loaders = {}
        for dsname in ['train', 'valid']:
            src = H5Dataset(src_h5, dsname).data[:, :max_length]
            tgt = H5Dataset(tgt_h5, dsname).data[:, :max_length + 1]
            self.max_length = \
                min(self.max_length + 1, src.shape[1], tgt[:,:-1].shape[1])
            srctgt = dask.array.concatenate((src, tgt[:,:-1]), axis=1)

            # Do not include BOS tokens in target output.
            tgt2 = tgt[:, 1:]

            # Shrink datasets if they are too large.
            if dsname == 'train':
                epoch_sz = epoch_size
                srctgt = srctgt[max_val_size:]
                tgt2 = tgt2[max_val_size:]
            elif dsname == 'valid' and max_val_size:
                epoch_sz = min(epoch_size, max_val_size)
                srctgt = srctgt[:max_val_size]
                tgt2 = tgt2[:max_val_size]
            self.datasets[dsname] = DaskDataset(srctgt, tgt2)

            if distributed:
                sampler = DistributedSubSampler(
                    self.datasets[dsname], world_size, pindex,
                    epoch_size=epoch_sz)
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
