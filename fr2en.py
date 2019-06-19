#!/usr/bin/env python
"""
This script launches model training in separate processes, one for each GPU.
"""

import fire
import os
import streamlit
import torch
import yaml

from train import build_learner, train_worker


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

    def _parse_args(self, config, gpu_ids=[], lr=None):
        """
        Parse the config file `config` and `gpu_ids` for various commands.
        """
        with open(config, 'r') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        default_config = params.get('default_config', None)
        if default_config:
            with open(default_config, 'r') as f:
                load_defaults(params, yaml.load(f, Loader=yaml.SafeLoader))

        if gpu_ids is not None:
            params['gpu_ids'] = gpu_ids
            if not isinstance(gpu_ids, list):
                gpu_ids = [gpu_ids]
        if 'optim' not in params:
            params['optim'] = {}
        if lr is not None:
            params['optim']['lr'] = lr

        project_dir = os.path.dirname(os.path.abspath(__file__))
        if 'model_name' not in params:
            raise ValueError('Expected parameter "model_name" not supplied.')

        return params, project_dir

    def train(self, config, gpu_ids=None, lr=None):
        """
        Train the model described in file `config` on GPUs `gpu_ids`.
        """
        params, project_dir = self._parse_args(config, gpu_ids, lr)
        if 'gpu_ids' not in params:
            # Training requires a GPU.
            raise ValueError('Expected parameter "gpu_ids" not supplied.')

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
        torch.multiprocessing.spawn(train_worker,
                                    args=(project_dir, params, comm_file),
                                    nprocs=len(params['gpu_ids']),
                                    join=True)

    def find_lr(self, config, gpu_id=0):
        """
        Search for an optimal learning rate and print plot to Streamlit.
        """
        print('************************************************************')
        print('TESTING LEARNING RATES: THIS WILL RUN FOR UNDER 100 BATCHES.')
        print('************************************************************\n')
        params, project_dir = self._parse_args(config, [gpu_id])
        params['gpu_ids'] = [gpu_id] if gpu_id is not None else []
        learn = build_learner(params, project_dir)

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
        learn = build_learner(params, project_dir)
        print(learn.summary())


if __name__ == '__main__':
    fire.Fire(PervasiveApp)
