"""
These classes implement a data loader for a dataset of pairs of translated
sentences together with their vocabularies.
"""

import h5py
import numpy as np
import pickle
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.utils.data  # Need this to avoid "module has no attribute 'data'".


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
                 distributed=False):
        self.model_name = model_name

        self.src_vocab = VocabData(src_infos)
        self.tgt_vocab = VocabData(tgt_infos)
        self.batch_size = batch_size
        self.max_length = max_length

        # Load HDF5 data file.
        self.datasets = {}
        self.loaders = {}
        with h5py.File(src_h5, 'r', libver='latest', swmr=True) as src_file:
            with h5py.File(tgt_h5, 'r', libver='latest', swmr=True) as tgt_file:
                for dsname in ['train', 'val', 'test']:
                    src = np.array(src_file[f'labels_{dsname}']).astype(
                        np.int64)
                    src = torch.as_tensor(src, dtype=torch.long)
                    src_bos = torch.tensor([[self.src_vocab.bos]],
                                           dtype=torch.int64).repeat(
                                               src.shape[0], 1)
                    src_eos = torch.tensor([[self.src_vocab.eos]],
                                           dtype=torch.int64).repeat(
                                               src.shape[0], 1)
                    src = torch.cat((src_bos, src, src_eos), 1)
                    tgt = np.array(tgt_file[f'labels_{dsname}']).astype(
                        np.int64)
                    tgt = torch.as_tensor(tgt, dtype=torch.long)
                    tgt_bos = torch.tensor([[self.src_vocab.bos]],
                                           dtype=torch.int64).repeat(
                                               tgt.shape[0], 1)
                    tgt_eos = torch.tensor([[self.src_vocab.eos]],
                                           dtype=torch.int64).repeat(
                                               tgt.shape[0], 1)
                    tgt = torch.cat((tgt_bos, tgt, tgt_eos), 1)
                    src_tgt = torch.cat((src, tgt), 1)
                    tgt2 = tgt.clone()
                    self.datasets[dsname] = torch.utils.data.TensorDataset(
                        src_tgt, tgt2)
                    # Hack to please fastai Learner.summary().
                    self.datasets[dsname].is_empty = False
                    if distributed:
                        sampler = DistributedSampler(self.datasets[dsname])
                    else:
                        sampler = None
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
        self.data_info = {}

        self.h5_filename = h5_filename
        self.dsname = dsname
        with h5py.File(self.h5_filename, 'r', libver='latest',
                       swmr=True) as h5_file:
            self.data_cache = torch.tensor(
                np.array(h5_file[dsname]).astype(np.int32))

    def __getitem__(self, index):
        """
        Get row `index` from the dataset.
        """
        return self.data_cache[index]

    def __len__(self):
        """
        Get the number of rows in the dataset.
        """
        return len(self.data_cache)
