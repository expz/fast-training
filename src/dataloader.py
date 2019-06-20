"""
These classes implement a data loader for a dataset of pairs of translated
sentences together with their vocabularies.
"""

import h5py
import math
import numpy as np
import pickle
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler
import torch.utils.data  # Need this to avoid "module has no attribute 'data'".


class DistributedSubSampler(DistributedSampler):
    """Sampler that extends the standard DistributedSampler to generate
    epochs of a fixed size from a larger dataset.

    This is useful to reduce epoch size for large datasets so that
    checkpoints will be reached more often. Attempting to put checkpoint
    callbacks in the middle of an epoch does not mesh well with the
    architecture of Pytorch/Fast.ai.

    Adapted from https://github.com/pytorch/pytorch/
                         blob/master/torch/utils/data/distributed.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, epoch_size=None):
        super().__init__(dataset, num_replicas, rank)
        if epoch_size is None:
            epoch_size = self.total_size
        else:
            self.num_samples = int(
                math.ceil(epoch_size * 1.0 / self.num_replicas))
            self.epoch_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """
        Returns this worker's portion of a sample of size `self.epoch_size` from
        the shuffled dataset.
        """
        # Deterministically shuffle based on epoch.
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        # TODO: Sample epochs without replacement.
        indices = indices[:self.epoch_size]

        # Add extra samples to make it evenly divisible.
        indices += indices[:(self.epoch_size - len(indices))]
        assert len(indices) == self.epoch_size

        # Subsample portion for this worker.
        indices = indices[self.rank:self.epoch_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class VocabData(object):

    def __init__(self, infos_fn):
        # Load index to word mapping from .infos file.
        self.infos_filename = infos_fn
        with open(self.infos_filename, 'rb') as f:
            infos = pickle.load(f, encoding='iso-8859-1')
        self.ix_to_word = infos['itow']
        self.vocab_size = len(self.ix_to_word)

        # Word to index mapping and special tokens.
        word_to_ix = {w: ix for ix, w in self.ix_to_word.items()}
        self.pad = word_to_ix['<PAD>']
        self.unk = word_to_ix['<UNK>']
        self.eos = word_to_ix['<EOS>']
        self.bos = word_to_ix['<BOS>']


class PervasiveDataLoader(object):
    """
    This data loader assumes that the data is stored in two HDF5 files
    both containing datasets named 'train', 'val' and 'test'.

    It encapsulates three Pytorch `TensorDataset` objects for training,
    validation and testing. It also contains three `DataLoader` objects,
    one for each dataset.

    Adapted from https://github.com/elbayadm/attn2d/blob/master/nmt/loader/dataloader.py
    """

    def __init__(self,
                 src_infos,
                 src_h5,
                 tgt_infos,
                 tgt_h5,
                 batch_size,
                 max_length,
                 model_name,
                 epoch_size=None,
                 max_val_size=None,
                 max_test_size=None,
                 distributed=False):

        def prepare_ds(h5fn, dsname, vocab, max_length):
            h5ds = H5Dataset(h5fn, f'labels_{dsname}')
            lens = H5Dataset(h5fn, f'lengths_{dsname}')
            bos = torch.tensor([[vocab.bos]], dtype=torch.int64).repeat(h5ds.data.shape[0], 1)
            pad = torch.tensor([[vocab.pad]], dtype=torch.int64).repeat(h5ds.data.shape[0], 1)
            ds = torch.cat((bos, h5ds.data[:, :max_length], pad), 1)
            indices = (torch.tensor(range(ds.shape[0])), lens.data.clamp(0, max_length))
            ds.index_put_(indices, torch.tensor([vocab.eos], dtype=torch.int64))
            return ds[:-1]

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        self.src_vocab = VocabData(src_infos)
        self.tgt_vocab = VocabData(tgt_infos)

        # Load HDF5 data file.
        self.datasets = {}
        self.loaders = {}
        for dsname in ['train', 'val', 'test']:
            src = prepare_ds(src_h5, dsname, self.src_vocab, self.max_length)
            tgt = prepare_ds(tgt_h5, dsname, self.tgt_vocab, self.max_length)
            srctgt = torch.cat((src, tgt), 1)
            tgt2 = tgt[:, 1:].clone()  # Do not include BOS tokens in target output.
            # Shrink datasets if they are too large.
            if dsname == 'val' and max_val_size:
                srctgt = srctgt[:max_val_size]
                tgt2 = tgt2[:max_val_size]
            elif dsname == 'test' and max_test_size:
                srctgt = srctgt[:max_test_size]
                tgt2 = tgt2[:max_test_size]
            self.datasets[dsname] = torch.utils.data.TensorDataset(srctgt, tgt2)
            # Hack to please fastai Learner.summary().
            self.datasets[dsname].is_empty = False
            # Define dataloader with distributed sampler.
            if distributed:
                sampler = DistributedSubSampler(self.datasets[dsname],
                                                epoch_size=epoch_size)
            else:
                # TODO: Define a non-distributed sub-sampler
                #       without replacement.
                sampler = RandomSampler(self.datasets[dsname],
                                        replacement=True,
                                        num_samples=epoch_size)
            self.loaders[dsname] = DataLoader(
                self.datasets[dsname],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                sampler=sampler)


class H5Dataset(torch.utils.data.Dataset):
    """
    This represents a dataset stored in an HDF5 file.
    """

    def __init__(self, h5_filename, dsname):
        """
        Loads dataset `dsname` from file `h5_filename` into a Pytorch tensor.
        """
        super().__init__()

        self.h5_filename = h5_filename
        self.dsname = dsname
        with h5py.File(self.h5_filename, 'r', libver='latest',
                       swmr=True) as h5_file:
            self.data = torch.as_tensor(
                np.array(h5_file[dsname]).astype(np.int64), dtype=torch.long)

    def __getitem__(self, index):
        """
        Get row `index` from the dataset.
        """
        return self.data[index]

    def __len__(self):
        """
        Get the number of rows in the dataset.
        """
        return len(self.data)
