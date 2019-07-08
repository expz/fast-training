"""
These functions implement machine translation model training.
"""

from datetime import datetime
from fastai.basic_data import DataBunch
from fastai.callbacks import LearnerCallback, SaveModelCallback
from fastai.callbacks.tensorboard import LearnerTensorboardWriter
from fastai.train import Learner
from fastprogress.fastprogress import format_time
from functools import partial
import logging
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from bleu import bleu_score
from dataloader import PervasiveDataLoader
from evaluate import beam_search
from pervasive import (
    Pervasive, PervasiveBert, PervasiveEmbedding, PervasiveOriginal, dilate,
    PervasiveDownsample
)
from vocab import VocabData


logger = logging.getLogger('fr2en')

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


def scaled_mse_loss(y, y_hat):
    """
    MSE loss scaled so that it usually lies in 0.1 - 100 range. This cannot
    be converted to a lambda, because it needs to be pickleable.
    """
    return 10000 * F.mse_loss(y, y_hat)


def build_learner(params, project_dir, pindex=0, comm_file=None, queues=None):
    """
    Builds a fastai `Learner` object containing the model and data specified by
    `params`. It is configured to run on GPU `device_id`. Assumes it is GPU
    `pindex` of `world_size` total GPUs. In case more than one GPU is being
    used, a file named `comm_file` is used to communicate between processes.
    """
    # For user friendly error messages, check these parameters exist.
    check_params(params, [
        'cpu',
        'data.batch_size',
        'data.dir',
        'data.epoch_size',
        'data.max_length',
        'data.max_val_size',
        'data.src',
        'data.tgt',
        'data.vocab',
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
        'network.kernel_size',
    ])

    model_name = params['model_name']

    # Try to make the directory for saving models.
    model_dir = os.path.join(project_dir, 'model', model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Configure GPU/CPU device settings.
    cpu = params['cpu']
    gpu_ids = params['gpu_ids'] if not cpu else []
    world_size = len(gpu_ids) if len(gpu_ids) > 0 else 1
    distributed = world_size > 1
    if gpu_ids:
        device_id = gpu_ids[pindex]
        device = torch.device(device_id)
        torch.cuda.set_device(device_id)
    else:
        device_id = None
        device = torch.device('cpu')

    # If distributed, initialize inter-process communication using shared file.
    if distributed:
        torch.distributed.init_process_group(backend='nccl',
                                             world_size=world_size,
                                             rank=pindex,
                                             init_method=f'file://{comm_file}')

    # Load vocabulary.
    vocab_path = os.path.join(params['data']['dir'], params['data']['vocab'])
    vocab = VocabData(vocab_path)

    # Load data.
    src_l = params['data']['src']
    tgt_l = params['data']['tgt']
    loader = PervasiveDataLoader(
        os.path.join(params['data']['dir'], f'{src_l}.h5'),
        os.path.join(params['data']['dir'], f'{tgt_l}.h5'),
        vocab,
        vocab,
        params['data']['batch_size'] // world_size,
        params['data']['max_length'],
        epoch_size=params['data']['epoch_size'],
        max_val_size=params['data']['max_val_size'],
        distributed=distributed,
        world_size=world_size,
        pindex=pindex)

    # Define neural network.
    # Max length is 1 more than setting to account for BOS.
    if params['network']['type'] == 'pervasive-embeddings':
        model = PervasiveEmbedding(
            params['network']['block_sizes'],
            vocab.bos,
            loader.max_length,
            loader.max_length,
            loader.datasets['train'].arrays[0].shape[2],
            params['encoder']['embedding_dim'],
            params['encoder']['embedding_dropout'],
            params['network']['dropout'],
            params['decoder']['prediction_dropout'],
            params['network']['division_factor'],
            params['network']['growth_rate'],
            params['network']['bias'],
            params['network']['efficient'])
        # Rescale loss by 100 for easier display in training output.
        loss_func = scaled_mse_loss
    elif params['network']['type'] == 'pervasive-downsample':
        model = PervasiveDownsample(
            params['network']['block_sizes'],
            vocab.bos,
            loader.max_length,
            loader.max_length,
            loader.datasets['train'].arrays[0].shape[2],
            params['encoder']['embedding_dim'],
            params['encoder']['embedding_dropout'],
            params['network']['dropout'],
            params['decoder']['prediction_dropout'],
            params['network']['division_factor'],
            params['network']['growth_rate'],
            params['network']['bias'],
            params['network']['efficient'],
            params['network']['kernel_size'])
        # Rescale loss by 100 for easier display in training output.
        loss_func = F.cross_entropy
    elif params['network']['type'] == 'pervasive-bert':
        model = PervasiveBert(
            params['network']['block_sizes'],
            vocab.bos,
            loader.max_length,
            loader.max_length,
            params['encoder']['embedding_dim'],
            params['encoder']['embedding_dropout'],
            params['network']['dropout'],
            params['decoder']['prediction_dropout'],
            params['network']['division_factor'],
            params['network']['growth_rate'],
            params['network']['bias'],
            params['network']['efficient'],
            params['network']['kernel_size'])
        loss_func = F.cross_entropy
    elif params['network']['type'] == 'pervasive-original':
        model = PervasiveOriginal(
            params['network']['block_sizes'],
            len(vocab),
            vocab.bos,
            loader.max_length,
            loader.max_length,
            params['encoder']['embedding_dim'],
            params['encoder']['embedding_dropout'],
            params['network']['dropout'],
            params['decoder']['prediction_dropout'],
            params['network']['division_factor'],
            params['network']['growth_rate'],
            params['network']['bias'],
            params['network']['efficient'],
            params['network']['kernel_size'])
        loss_func = F.cross_entropy
    elif params['network']['type'] == 'pervasive':
        model = Pervasive(
            params['network']['block_sizes'],
            len(vocab),
            vocab.bos,
            loader.max_length,
            loader.max_length,
            params['encoder']['initial_emb_dim'],
            params['encoder']['embedding_dim'],
            params['encoder']['embedding_dropout'],
            params['network']['dropout'],
            params['decoder']['prediction_dropout'],
            params['network']['division_factor'],
            params['network']['growth_rate'],
            params['network']['bias'],
            params['network']['efficient'],
            params['network']['kernel_size'])
        loss_func = F.cross_entropy

    model.init_weights()
    if device_id is not None:
        if not torch.cuda.is_available():
            raise ValueError(
                'Request to train on GPU {device_id}, but not GPU found.')
        model.cuda(device_id)
        if distributed:
            model = DistributedDataParallel(model, device_ids=[device_id])
    data = DataBunch(loader.loaders['train'],
                     loader.loaders['valid'],
                     loader.loaders['valid'],
                     device=device)

    # Create Learner with Adam optimizer.
    learn = Learner(data, model, loss_func=loss_func, model_dir=model_dir)
    AdamP = partial(
        torch.optim.Adam,
        betas=(params['optim']['beta1'], params['optim']['beta2']))
    learn.opt_func = AdamP
    learn.wd = params['optim']['wd']

    return (
        learn, loader.loaders['train'].src_vocab,
        loader.loaders['train'].tgt_vocab)


def restore(learn, model_fn, do_dilate=False):
    """
    Restores the weights of a model saved to `model_fn` to the model of
    the Learner `learn`.
    """
    epoch = None
    if model_fn is not None:
        try:
            # Turning off `strict` means it is okay for the saved model not
            # to have weights for all the parameters of the current model.
            state = torch.load(model_fn, map_location=learn.data.device)
            model = learn.model
            if isinstance(model, DistributedDataParallel):
                model = model.module
            model.load_state_dict(state['model'], strict=False)
            if do_dilate:
                dilate(model.network, fill_with_avg=True)
        except FileNotFoundError:
            raise Exception(f'The model file {model_fn} was not found!')
        fields = model_fn.split('/')[-1].split('_')
        if len(fields) > 1:
            try:
                epoch = int(fields[1].split('.')[0]) + 1
            except ValueError:
                pass
    return epoch


def train_worker(pindex,
                 project_dir,
                 params,
                 comm_file=None,
                 restore_fn=None,
                 do_dilate=False,
                 queues=None):
    """
    Trains the model as specified by `params` on GPU `gpu_ids[pindex]`.
    Uses `comm_file` to communicate between processes.
    Saves models and event logs to subdirectories of `project_dir`.

    This is run in separate processes from the command line app, with
    one process per GPU.

    Optionally load a saved model with filename `restore`.
    """
    # Variable used for distributed processing.
    if not os.getenv('RANK', None):
        os.environ['RANK'] = str(pindex)

    learn, _, _ = build_learner(params, project_dir, pindex, comm_file, queues)

    # Restore saved model if necessary.
    epoch = restore(learn, restore_fn, do_dilate)

    learn.model.cuda(params['gpu_ids'][pindex])

    # Callbacks.
    logs_path = learn.path / 'logs'
    os.makedirs(f'{logs_path}/{params["model_name"]}', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    csv_fn = f'logs/{params["model_name"]}/log-{params["model_name"]}-{ts}'
    # TODO: Enabling Tensorboard metrics causes an error.
    # tbwriter = LearnerTensorboardWriter(learn, logs_path, params['model_name'])
    # tbwriter.metrics_root = 'metrics/'
    learn.callbacks = [
        # Save callback causes 'Model not found' error when restoring.
        SaveModelCallback(learn, every='epoch', name='model'),
        CSVLogger(learn, csv_fn),
    #    tbwriter,
    ]
    if params['network']['type'] != 'pervasive-embeddings':
        learn.metrics.append(BLEUScoreMetric(learn, 5, queues, pindex))

    if params['freeze']:
        if isinstance(learn.model, DistributedDataParallel):
            model = learn.model.module
        model = learn.model
        learn.split([model.unprojection, model.prediction_dropout])
        # Untie target language embedding weights from input layer.
        model.prediction.weight = torch.nn.Parameter(
            model.prediction.weight.clone())
        learn.freeze_to(1)

    # Train with a one cycle schedule for each epoch.
    check_params(params, [
        'optim.epochs',
        'optim.lr',
    ])
    if pindex == 0:
        g = len(params['gpu_ids']) if params['gpu_ids'] else 0
        logger.info(f"Learning rate: {params['optim']['lr']}, "
                    f"Beta1: {params['optim']['beta1']}, "
                    f"Beta2: {params['optim']['beta2']}, "
                    f"Weight decay: {params['optim']['wd']}, "
                    f"Batch size: {params['data']['batch_size']}, "
                    f"Epoch size: {params['data']['epoch_size']}, "
                    f"Epochs: {params['optim']['epochs']}, "
                    f"GPUs: {g}")
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
        """Saves the start time at the beginning of an epoch."""
        if self.add_time:
            self.start_epoch = time.time()

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
    """
    A BLEU score `Callback` that generates an output sentence using beam
    search with beam size `beam_size` and then calculates its BLEU score.
    """

    def __init__(self, learn, beam_size=5, queues=None, pindex=None):
        """
        `queues` is a list of Queues for passing BLEU scores between processes.
        `pindex` is the index of the current process which selects the
                 queue of the current process from `queues`.
        """
        super().__init__(learn)
        self.name = 'bleu'
        self.beam_size = beam_size
        self.tgt_vocab = learn.data.valid_dl.tgt_vocab
        if isinstance(learn.model, DistributedDataParallel):
            self.Ts = self.learn.model.module.Ts
            self.Tt = self.learn.model.module.Tt
        else:
            self.Ts = self.learn.model.Ts
            self.Tt = self.learn.model.Tt
        self.eos = self.tgt_vocab.eos
        self.pad = self.tgt_vocab.pad
        self.queues = queues
        self.pindex = pindex

    def on_epoch_begin(self, **kwargs):
        """
        Resets the BLEU score and sentence count at the beginning of each epoch.
        """
        self.bleu, self.count = 0.0, 0

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        """
        Calculates output sentence using beam search for every batch of
        validation examples.
        """
        if not train:
            batch_size = last_input.size(0)
            src_data, tgt_data = last_input.split([self.Ts, self.Tt], dim=1)
            out_data = beam_search(self.learn.model, src_data, self.beam_size,
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
        """
        Passes BLEU scores between process to calculate total average.
        """
        if not train:
            for i, q in enumerate(self.queues):
                if i != self.pindex:
                    q.put((self.count, self.bleu))
            qs = len(self.queues) - 1
            i = 0
            max_iter = 200
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
                logger.warn(f'process {self.pindex} did not receive bleu '
                            'scores from all sibling processes. bleu scores '
                            'will not reflect all data.')
            return {'last_metrics': last_metrics + [self.bleu / self.count]}
