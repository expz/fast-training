import os
import yaml


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


def parse_config(config,
                 device_ids=['cpu'],
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

    # Load the default config if any.
    default_config = params.get('default_config', None)
    if default_config:
        with open(default_config, 'r') as f:
            load_defaults(params, yaml.load(f, Loader=yaml.SafeLoader))

    # Save command line arguments to parameters
    if device_ids:
        if not isinstance(device_ids, list):
            device_ids = [device_ids]
        if 'cpu' in device_ids:
            params['cpu'] = True
            params['gpu_ids'] = None
            gpus = 0
        else:
            params['cpu'] = False
            params['gpu_ids'] = list(map(lambda s: int(s), device_ids))
            gpus = len(params['gpu_ids'])
    if 'optim' not in params:
        params['optim'] = {}
    if lr is not None:
        params['optim']['lr'] = float(lr)
    if epochs is not None:
        params['optim']['epochs'] = int(epochs)
    if 'data' not in params:
        params['data'] = {}
    if batch_size is not None:
        params['data']['batch_size'] = \
            int(batch_size) * gpus if gpus else int(batch_size)
    if epoch_size is not None:
        params['data']['epoch_size'] = int(epoch_size)

    # Set default values for these parameters.
    if 'max_val_size' not in params['data']:
        params['data']['max_val_size'] = None
    if 'loader' not in params['data']:
        params['data']['loader'] = 'standard'
    if freeze is not None or 'freeze' not in params:
        freeze = freeze if freeze is not None else False
        params['freeze'] = freeze

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if 'model_name' not in params:
        raise ValueError('Expected parameter "model_name" not supplied.')

    return params, project_dir
