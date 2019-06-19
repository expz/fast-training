"""
These functions implement machine translation model training.
"""

from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
import fastai.train  # Required to add lr_find() function to Learner.
import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from dataloader import PervasiveDataLoader
from pervasive import Pervasive


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


def save_parallel(learn, name):
    """
    Save a model. Works also for parallelized models.

    From https://forums.fast.ai/t/how-to-use-multiple-gpus/26057/10?u=shaun1
    """
    learn.save(name)
    state_dict = torch.load(f'{name}.pth')

    # Create new OrderedDict that does not contain `module`.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # Remove `module`.
        new_state_dict[name] = v
    # Load params.
    model.load_state_dict(new_state_dict)
    learn.save(name)
    learn.save_encoder(f'{name}_encoder')


def build_learner(params,
                  project_dir,
                  pindex=0,
                  comm_file=None):
    """
    Builds a fastai `Learner` object containing the model and data specified by
    `params`. It is configured to run on GPU `device_id`. Assumes it is GPU
    `pindex` of `world_size` total GPUs. In case more than one GPU is being
    used, a file named `comm_file` is used to communicate between processes.
    """
    model_name = params['model_name']
    model_dir = os.path.join(project_dir, 'model')
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
        'data.max_length',
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
    loader = PervasiveDataLoader(src_infos,
                                 src_h5,
                                 tgt_infos,
                                 tgt_h5,
                                 batch_size,
                                 params['data']['max_length'],
                                 model_name,
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
    # Max length is 2 more than setting to account for BOS and EOS.
    model = Pervasive(
        model_name, loader.src_vocab, loader.tgt_vocab,
        params['network']['block_sizes'], params['data']['max_length'] + 2,
        params['data']['max_length'] + 2, params['encoder']['embedding_dim'],
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
    return Learner(data, model, loss_func=F.cross_entropy, model_dir=model_dir)


def train_worker(pindex, project_dir, params, comm_file=None):
    """
    Trains the model as specified by `params` on GPU `gpu_ids[pindex]`.
    Uses `comm_file` to communicate between processes.
    Saves models and event logs to subdirectories of `project_dir`.

    This is run in separate processes from the command line app, with
    one process per GPU.
    """
    # Variable used for distributed processing.
    if not os.getenv('RANK', None):
        os.environ['RANK'] = str(pindex)

    learn = build_learner(params, project_dir, pindex, comm_file)
    lr = params['optim']['lr']
    learn.fit_one_cycle(1, lr, tot_epochs=1)  # One epoch.
