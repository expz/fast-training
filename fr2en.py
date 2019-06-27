#!/usr/bin/env python
"""
This script launches model training in separate processes, one for each GPU.
"""

import fire
import multiprocessing
import os
import streamlit
import torch
from torch.nn.parallel import DistributedDataParallel
import yaml

from train import build_learner, build_learner_without_data, train_worker
from evaluate import beam_search


def load_defaults(params, defaults):
    """
    A configuration file can specify another config file with default values.
    This functions loads the default values into the `params` dictionary
    without overwriting existing values.
    """
    for key in defaults:
        if isinstance(defaults[key], dict):
            if key not in params:
                params[key] = {}
            load_defaults(params[key], defaults[key])
        else:
            if key not in params:
                params[key] = defaults[key]


class PervasiveApp(object):
    """
    This is the command line app that the `fire` packages exposes
    using command line arguments.
    """

    def _parse_args(self,
                    config,
                    device_ids=None,
                    lr=None,
                    batch_size=None,
                    epochs=None,
                    epoch_size=None,
                    freeze=None):
        """
        Parse the config file `config` and `gpu_ids` for various commands.
        """
        with open(config, 'r') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        default_config = params.get('default_config', None)
        if default_config:
            with open(default_config, 'r') as f:
                load_defaults(params, yaml.load(f, Loader=yaml.SafeLoader))

        if device_ids is not None:
            if not isinstance(device_ids, list):
                device_ids = [device_ids]
            if 'cpu' in device_ids:
                params['cpu'] = True
                params['gpu_ids'] = None
            else:
                params['cpu'] = False
                params['gpu_ids'] = list(map(lambda s: int(s), device_ids))
        if 'optim' not in params:
            params['optim'] = {}
        if lr is not None:
            params['optim']['lr'] = float(lr)
        if epochs is not None:
            params['optim']['epochs'] = int(epochs)
        if batch_size is not None:
            params['data']['batch_size'] = int(batch_size)
        if epoch_size is not None:
            params['data']['epoch_size'] = int(epoch_size)
        # Set default values for these parameters.
        if 'max_val_size' not in params['data']:
            params['data']['max_val_size'] = None
        if 'max_test_size' not in params['data']:
            params['data']['max_test_size'] = None
        if 'loader' not in params['data']:
            params['data']['loader'] = 'standard'
        if freeze is not None or 'freeze' not in params:
            freeze = freeze if freeze is not None else False
            params['freeze'] = freeze

        project_dir = os.path.dirname(os.path.abspath(__file__))
        if 'model_name' not in params:
            raise ValueError('Expected parameter "model_name" not supplied.')

        return params, project_dir

    def _restore(self, learn, filename):
        """
        Load the model saved at `filename` if it exists.
        """
        if filename is not None:
            try:
                # Remove extension if provided to match `load()`'s expectation.
                if filename[-4:] == '.pth':
                    filename = filename[:-4]
                learn.load(filename, purge=False)
                print(f'Loaded model {filename}.')
            except FileNotFoundError:
                print(f'The model file {learn.model_dir}/{filename}.pth '
                      'was not found!')
                return

    def train(self,
              config,
              device_ids=None,
              lr=None,
              checkpoint=None,
              restore=None,
              batch=None,
              epochs=None,
              epoch_size=None,
              freeze=False):
        """
        Train the model described in file `config` on devices `device_ids`.
        """
        params, project_dir = self._parse_args(config, device_ids, lr, batch,
                                               epochs, epoch_size, freeze)

        # Prepare a place for the shared process communication file.
        model_name = params['model_name']
        comm_file = f'{project_dir}/model/{model_name}/pgroup_shared'
        os.makedirs(f'{project_dir}/model/{model_name}', exist_ok=True)
        try:
            os.remove(comm_file)
        except FileNotFoundError:
            pass

        # Variables used for distributed processing.
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '3892'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
        m = multiprocessing.Manager()
        nprocs = max(1, len(params['gpu_ids']))
        qs = [m.Queue() for _ in range(nprocs)]
        torch.multiprocessing.spawn(train_worker,
                                    args=(project_dir, params, comm_file,
                                          checkpoint, restore, qs),
                                    nprocs=nprocs,
                                    join=True)

    def example(self, config, gpu_id=0, restore=None, batch=None):
        """
        Print a list of `batch` many example translations. This function
        requires the `restore` argument to be most useful. `restore` should be
        the path of the saved model relative to the current model's folder.
        The current model is specified by the `name` field in the config.
        """
        params, project_dir = \
            self._parse_args(config, [gpu_id], batch_size=batch)
        learn, src_vocab, tgt_vocab = build_learner(params, project_dir)
        self._restore(learn, restore)
        batch, tgt = next(iter(learn.data.valid_dl))
        src_data, tgt_data = \
            batch.split([learn.model.Ts, learn.model.Tt], dim=1)
        src_text = src_vocab.to_text(src_data)
        out_data = beam_search(learn, src_data, 5, params['data']['max_length'])
        out_text = tgt_vocab.to_text(out_data)
        for src, out in zip(src_text, out_text):
            print(f'IN: {src}')
            print(f'OUT: {out}')

    def translate(self, config, src_text, gpu_id=0, restore=None):
        """
        Translates a single sentence using the model saved at `restore`.

        TODO: For better performance, this should parse `src_text` using
              byte-pair encoding (BPE).
        """
        params, project_dir = \
            self._parse_args(config, [gpu_id], batch_size=1)
        learn, src_vocab, tgt_vocab = \
            build_learner_without_data(params, project_dir)
        self._restore(learn, restore)
        src_data = torch.tensor([
            src_vocab.to_ints(src_text, params['data']['max_length'] + 1)
        ]).to(torch.device(gpu_id))
        out_data = beam_search(learn, src_data, 5, params['data']['max_length'])
        out_text = tgt_vocab.to_text(out_data)[0]
        print(out_text)

    def find_lr(self, config, gpu_id=0):
        """
        Search for an optimal learning rate and print plot to Streamlit.
        """
        print('************************************************************')
        print('TESTING LEARNING RATES: THIS WILL RUN FOR UNDER 100 BATCHES.')
        print('************************************************************\n')
        params, project_dir = self._parse_args(config, [gpu_id])
        params['gpu_ids'] = [gpu_id] if gpu_id is not None else []
        learn, _, _ = build_learner(params, project_dir)

        streamlit.title('Find the best learning rate')
        streamlit.header(f'Model {params["model_name"]}')
        streamlit.text(
            'Choose the learning rate where the graph has its steepest decline.'
        )

        learn.lr_find()
        learn.recorder.plot(return_fig=True)
        streamlit.pyplot()

    def summary(self, config):
        """
        Print a summary of the model architecture described by file `config`.
        """
        params, project_dir = self._parse_args(config)
        learn, _, _ = build_learner(params, project_dir)
        print(learn.summary())


if __name__ == '__main__':
    fire.Fire(PervasiveApp)
