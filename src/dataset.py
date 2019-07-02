import dask.array
import h5py
import numpy as np
import torch
import torch.utils.data


class DaskDataset(torch.utils.data.Dataset):

    def __init__(self, *arrays):
        # `is_empty` is a hack to please fastai Learner.summary().
        self.is_empty = not (len(arrays) and arrays[0].shape[0])
        self.dtypes = tuple(self._torch_dtype(array.dtype) for array in arrays)
        self.arrays = arrays

    @classmethod
    def _torch_dtype(cls, dask_dtype):
        if dask_dtype == np.float16:
            return torch.float16
        elif dask_dtype == np.float32:
            return torch.float32
        elif dask_dtype == np.float64:
            return torch.float64
        elif dask_dtype == np.int32:
            return torch.int32
        elif dask_dtype == np.int64:
            return torch.int64
        else:
            raise NotImplementedError(
                f'Datatype {dask_dtype} not supported by DaskDataset.')

    def __getitem__(self, index):
        return tuple(
            torch.tensor(np.array(array[index]), dtype=self.dtypes[i])
            for i, array in enumerate(self.arrays))

    def __len__(self):
        return self.arrays[0].shape[0] if self.arrays else 0


class H5Dataset(torch.utils.data.Dataset):
    """
    This represents a dataset stored in an HDF5 file.
    """

    def __init__(self, h5fn, dsname):
        """
        Loads dataset `dsname` from file `h5_filename` into a Pytorch tensor.
        """
        super().__init__()

        self.h5fn = h5fn
        self.dsname = dsname
        self.h5file = h5py.File(self.h5fn, 'r', libver='latest', swmr=True)
        dtype_name = self.h5file[dsname].attrs['dtype']
        if dtype_name == 'int64':
            self.dtype = torch.int64
        elif dtype_name == 'int32':
            self.dtype = torch.int64  # Model seems to want Long's.
        elif dtype_name == 'float64':
            self.dtype = torch.float64
        elif dtype_name == 'float32':
            self.dtype = torch.float32
        elif dtype_name == 'float16':
            self.dtype = torch.float16
        else:
            raise NotImplementedError(
                f'Dataset datatype {dtype_name} not supported.')
        ch_sz = ('auto', self.h5file[dsname].shape[1])
        if len(self.h5file[dsname].shape) > 2:
            ch_sz = ch_sz + (self.h5file[dsname].shape[2],)
        self.data = dask.array.from_array(self.h5file[dsname], chunks=ch_sz)

    def __getitem__(self, index):
        """Get row `index` from the dataset."""
        return torch.tensor(np.array(self.data[index]), dtype=self.dtype)

    def __len__(self):
        """Get the number of rows in the dataset."""
        return len(self.data)

    def get_batch(self, indices):
        """Get a batch of rows corresponding to `indices`."""
        return torch.tensor(np.array(self.data[indices]), dtype=self.dtype)
        
    def size(self, dim):
        return self.data.shape[dim]