"""
These functions implement machine translation model training.
"""

from collections import OrderedDict
from datetime import datetime
from fastai.basic_data import DataBunch
from fastai.callbacks import LearnerCallback, SaveModelCallback
from fastai.callbacks.tensorboard import LearnerTensorboardWriter
from fastai.train import validate, Learner
from fastprogress.fastprogress import format_time
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from bleu import bleu_score
from dataloader import (PervasiveDataLoader, ProgressivePervasiveDataLoader,
                        VocabData)
from evaluate import beam_search
from pervasive import Pervasive

src_dir = os.path.dirname(os.path.abspath(__file__))


def check_params(params, param_list):
    """
    Checks that a list of parameters is found in the config file
    and throws an exception if not.
    """
    for param in param_list:
        try:
            val = params
            for key in param.split('.'):
                val = val[key]
        except (KeyError, TypeError):
            raise ValueError(f'Expected parameter "{param}" not supplied.')


def build_learner_without_data(params, project_dir):
    model_name = params['model_name']
    model_dir = os.path.join(project_dir, 'model', model_name)
    try:
        # Try to make the directory for saving models.
        os.makedirs(model_dir)
    except FileExistsError:
        pass

    # Configure GPU/CPU device settings.
    gpu_ids = params['gpu_ids']
    if gpu_ids:
        device_id = gpu_ids[0]
        device_name = torch.cuda.get_device_name(device_id)
        device = torch.device(device_id)
        torch.cuda.set_device(device_id)
    else:
        device_id = None
        device_name = 'cpu'
        device = torch.device('cpu')

    data_dir = params['data']['dir']
    src_l = params['data']['src']
    tgt_l = params['data']['tgt']
    src_vocab = VocabData(os.path.join(data_dir, f'{src_l}.infos'))
    tgt_vocab = VocabData(os.path.join(data_dir, f'{tgt_l}.infos'))

    # Define neural network.
    check_params(params, [
        'decoder.embedding_dim',
        'decoder.embedding_dropout',
        'decoder.prediction_dropout',
        'encoder.embedding_dim',
        'encoder.embedding_dropout',
        'network.bias',
        'network.block_sizes',
        'network.division_factor',
        'network.dropout',
        'network.efficient',
        'network.growth_rate',
    ])
    # Max length is 1 more than setting to account for BOS.
    model = Pervasive(
        model_name, src_vocab, tgt_vocab, params['network']['block_sizes'],
        params['data']['max_length'] + 1, params['data']['max_length'] + 1,
        params['encoder']['embedding_dim'], params['decoder']['embedding_dim'],
        params['encoder']['embedding_dropout'], params['network']['dropout'],
        params['decoder']['embedding_dropout'],
        params['decoder']['prediction_dropout'],
        params['network']['division_factor'], params['network']['growth_rate'],
        params['network']['bias'], params['network']['efficient'])

    model.init_weights()
    dummy = torch.tensor([])
    if device_id is not None:
        if not torch.cuda.is_available():
            raise ValueError(
                'Request to train on GPU {device_id}, but not GPU found.')
        model.cuda(device_id)
        dummy.cuda(device_id)
    dataset = torch.utils.data.TensorDataset(dummy, dummy)
    data = DataBunch(DataLoader(dataset),
                     DataLoader(dataset),
                     DataLoader(dataset),
                     device=device)
    learn = Learner(data, model, loss_func=F.cross_entropy, model_dir=model_dir)
    return learn, src_vocab, tgt_vocab


def build_learner(params, project_dir, pindex=0, comm_file=None, queues=None):
    """
    Builds a fastai `Learner` object containing the model and data specified by
    `params`. It is configured to run on GPU `device_id`. Assumes it is GPU
    `pindex` of `world_size` total GPUs. In case more than one GPU is being
    used, a file named `comm_file` is used to communicate between processes.
    """
    model_name = params['model_name']
    model_dir = os.path.join(project_dir, 'model', model_name)
    try:
        # Try to make the directory for saving models.
        os.makedirs(model_dir)
    except FileExistsError:
        pass

    # Configure GPU/CPU device settings.
    gpu_ids = params['gpu_ids']
    world_size = len(gpu_ids) if len(gpu_ids) > 0 else 1
    distributed = world_size > 1
    if gpu_ids:
        device_id = gpu_ids[pindex]
        device_name = torch.cuda.get_device_name(device_id)
        device = torch.device(device_id)
        torch.cuda.set_device(device_id)
    else:
        device_id = None
        device_name = 'cpu'
        device = torch.device('cpu')

    # If distributed, initialize inter-process communication using shared file.
    if distributed:
        torch.distributed.init_process_group(backend='nccl',
                                             world_size=world_size,
                                             rank=pindex,
                                             init_method=f'file://{comm_file}')

    # Load data.
    check_params(params, [
        'data.batch_size',
        'data.dir',
        'data.epoch_size',
        'data.max_length',
        'data.max_test_size',
        'data.max_val_size',
        'data.src',
        'data.tgt',
    ])
    batch_size = params['data']['batch_size'] // world_size
    data_dir = params['data']['dir']
    src_l = params['data']['src']
    tgt_l = params['data']['tgt']
    src_infos = os.path.join(data_dir, f'{src_l}.infos')
    tgt_infos = os.path.join(data_dir, f'{tgt_l}.infos')
    src_h5 = os.path.join(data_dir, f'{src_l}.h5')
    tgt_h5 = os.path.join(data_dir, f'{tgt_l}.h5')
    if params['data']['loader'] == 'progressive':
        loader = ProgressivePervasiveDataLoader(
            src_infos,
            src_h5,
            tgt_infos,
            tgt_h5,
            batch_size,
            params['data']['max_length'],
            model_name,
            epoch_size=params['data']['epoch_size'],
            max_val_size=params['data']['max_val_size'],
            max_test_size=params['data']['max_test_size'],
            distributed=distributed)
        params['data']['max_length'] = params['data']['max_length'] // 2
    else:
        loader = PervasiveDataLoader(
            src_infos,
            src_h5,
            tgt_infos,
            tgt_h5,
            batch_size,
            params['data']['max_length'],
            model_name,
            epoch_size=params['data']['epoch_size'],
            max_val_size=params['data']['max_val_size'],
            max_test_size=params['data']['max_test_size'],
            distributed=distributed)
    # Define neural network.
    check_params(params, [
        'decoder.embedding_dim',
        'decoder.embedding_dropout',
        'decoder.prediction_dropout',
        'encoder.embedding_dim',
        'encoder.embedding_dropout',
        'network.bias',
        'network.block_sizes',
        'network.division_factor',
        'network.dropout',
        'network.efficient',
        'network.growth_rate',
    ])
    # Max length is 1 more than setting to account for BOS.
    model = Pervasive(
        model_name, loader.src_vocab, loader.tgt_vocab,
        params['network']['block_sizes'], params['data']['max_length'] + 1,
        params['data']['max_length'] + 1, params['encoder']['embedding_dim'],
        params['decoder']['embedding_dim'],
        params['encoder']['embedding_dropout'], params['network']['dropout'],
        params['decoder']['embedding_dropout'],
        params['decoder']['prediction_dropout'],
        params['network']['division_factor'], params['network']['growth_rate'],
        params['network']['bias'], params['network']['efficient'])
    model.init_weights()
    if device_id is not None:
        if not torch.cuda.is_available():
            raise ValueError(
                'Request to train on GPU {device_id}, but not GPU found.')
        model.cuda(device_id)
        if distributed:
            model = DistributedDataParallel(model, device_ids=[device_id])
    data = DataBunch(loader.loaders['train'],
                     loader.loaders['val'],
                     loader.loaders['test'],
                     device=device)
    learn = Learner(data, model, loss_func=F.cross_entropy, model_dir=model_dir)
    return learn, loader.src_vocab, loader.tgt_vocab


def train_worker(pindex,
                 project_dir,
                 params,
                 comm_file=None,
                 checkpoint=None,
                 restore=None,
                 queues=None):
    """
    Trains the model as specified by `params` on GPU `gpu_ids[pindex]`.
    Uses `comm_file` to communicate between processes.
    Saves models and event logs to subdirectories of `project_dir`.

    This is run in separate processes from the command line app, with
    one process per GPU.

    Optionally save the model every `checkpoint` batches.

    Optionally load a saved model with filename `restore`.
    """
    # Variable used for distributed processing.
    if not os.getenv('RANK', None):
        os.environ['RANK'] = str(pindex)

    learn, _, _ = build_learner(params, project_dir, pindex, comm_file, queues)

    # Restore saved model if necessary.
    epoch = None
    if restore is not None:
        try:
            # Remove extension if provided to match `load()`'s expectation.
            if restore[-4:] == '.pth':
                restore = restore[:-4]
            learn.load(restore, purge=False)
        except FileNotFoundError:
            if pindex == 0:
                # Only print once even with mulitple workers.
                print(f'The model file {learn.model_dir}/{restore}.pth '
                      'was not found!')
            return
        fields = restore.split('/')[-1].split('_')
        if len(fields) > 1:
            try:
                epoch = int(fields[1]) + 1
            except:
                pass
    # Callbacks.
    logs_path = learn.path / 'logs'
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    csv_fn = f'logs/{params["model_name"]}/log-{params["model_name"]}-{ts}'
    tbwriter = LearnerTensorboardWriter(learn, logs_path, params['model_name'])
    tbwriter.metrics_root = 'metrics/'
    learn.callbacks = [
        SaveModelCallback(learn, every='epoch', name='model'),
        CSVLogger(learn, csv_fn),
        tbwriter,
    ]
    learn.metrics.append(BLEUScoreMetric(learn, 5, queues, pindex))

    if params['freeze']:
        if isinstance(learn.model, DistributedDataParallel):
            model = learn.model.module
        model = learn.model
        learn.split([model.linear, model.prediction_dropout])
        # Untie target language embedding weights from input layer.
        model.prediction.weight = torch.nn.Parameter(prediction.weight.clone())
        learn.freeze_to(1)

    # Train with a one cycle schedule for each epoch.
    check_params(params, [
        'optim.epochs',
        'optim.lr',
    ])
    learn.fit_one_cycle(params['optim']['epochs'],
                        params['optim']['lr'],
                        tot_epochs=params['optim']['epochs'],
                        start_epoch=epoch)


class CSVLogger(LearnerCallback):
    """
    A `LearnerCallback` that saves history of metrics while training
    `learn` into CSV `filename`.

    This is adapted from the fastai library. It is copied here so the file
    writes can be written using `with` blocks. This (1) forces the files to
    flush the log after every write (2) allows multiple processes to write
    to the file in the distributed training setting. Original:

    https://github.com/fastai/fastai/blob/master/fastai/callbacks/csv_logger.py
    """

    def __init__(self,
                 learn: Learner,
                 filename: str = 'history',
                 append: bool = False):
        super().__init__(learn)
        self.filename, self.append = filename, append
        self.path = self.learn.path / f'{filename}.csv'
        self.add_time = True

    def read_logged_file(self):
        "Read the content of saved file"
        return pd.read_csv(self.path)

    def on_train_begin(self, **kwargs):
        "Prepare file with metric names."
        self.path.parent.mkdir(parents=True, exist_ok=True)
        names = self.learn.recorder.names[:(None if self.add_time else -1)]
        header = ','.join(names) + '\n'
        if self.append:
            with self.path.open('a') as f:
                f.write(header)
        else:
            with self.path.open('w') as f:
                f.write(header)

    def on_epoch_begin(self, **kwargs):
        if self.add_time: self.start_epoch = time.time()

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        last_metrics = last_metrics if last_metrics is not None else []
        metrics = zip(self.learn.recorder.names,
                      [epoch, smooth_loss] + last_metrics)
        stats = [
            str(stat) if isinstance(stat, int) else
            '#na#' if stat is None else f'{stat:.6f}' for name, stat in metrics
        ]
        if self.add_time:
            stats.append(format_time(time.time() - self.start_epoch))
        str_stats = ','.join(stats)
        with self.path.open('a') as f:
            f.write(str_stats + '\n')


class BLEUScoreMetric(LearnerCallback):

    def __init__(self, learn, beam_size=5, queues=None, pindex=None):
        super().__init__(learn)
        self.name = 'bleu'
        self.beam_size = beam_size
        if isinstance(learn.model, DistributedDataParallel):
            self.tgt_vocab = learn.model.module.tgt_vocab
            self.Ts = self.learn.model.module.Ts
            self.Tt = self.learn.model.module.Tt
        else:
            self.tgt_vocab = learn.model.tgt_vocab
            self.Ts = self.learn.model.Ts
            self.Tt = self.learn.model.Tt
        self.eos = self.tgt_vocab.eos
        self.pad = self.tgt_vocab.pad
        self.queues = queues
        self.pindex = pindex

    def on_epoch_begin(self, **kwargs):
        self.bleu, self.count = 0.0, 0

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if not train:
            batch_size = last_input.size(0)
            src_data, tgt_data = last_input.split([self.Ts, self.Tt], dim=1)
            out_data = beam_search(self.learn, src_data, self.beam_size,
                                   self.Tt - 1)
            assert (list(out_data.shape) == [batch_size, self.Tt - 1])
            bleu = 0.0
            for b in range(batch_size):
                out_l = []
                for i in range(self.Tt - 1):
                    if (out_data[b][i].item() == self.eos or
                            out_data[b][i].item() == self.pad):
                        break
                    out_l.append(str(out_data[b][i].item()))
                tgt_l = []
                for i in range(1, self.Tt):
                    if (tgt_data[b][i].item() == self.eos or
                            tgt_data[b][i].item() == self.pad):
                        # The Moses BLEU score script gives 0 for sentences of
                        # length less than four, so ignore those BLEU score.
                        if i < 4:
                            batch_size -= 1
                        break
                    tgt_l.append(str(tgt_data[b][i].item()))
                bleu += bleu_score([' '.join(out_l)], [[' '.join(tgt_l)]]) * 100
            self.count += batch_size
            self.bleu += bleu

    def on_epoch_end(self, last_metrics, train, **kwargs):
        if not train:
            for i, q in enumerate(self.queues):
                if i != self.pindex:
                    q.put((self.count, self.bleu))
            qs = len(self.queues) - 1
            i = 0
            max_iter = 20
            while qs > 0 and i < max_iter:
                while not self.queues[self.pindex].empty():
                    c, b = self.queues[self.pindex].get()
                    self.count += c
                    self.bleu += b
                    qs -= 1
                if qs > 0:
                    time.sleep(0.1)
                    i += 1
            if i == max_iter:
                print(f'WARNING: process {self.pindex} did not receive bleu '
                      'scores from all sibling processes. bleu scores will '
                      'not reflect all data.')
            return {'last_metrics': last_metrics + [self.bleu / self.count]}
