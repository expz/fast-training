import argparse
import os
import sys
import torch

from config import parse_config
from evaluate import beam_search
from pervasive import Pervasive
from vocab import VocabData


def translate(src_text, config, model_path):
    params, project_dir = \
        parse_config(config, batch_size=1)
    model, src_vocab, tgt_vocab = build_model(params, project_dir)
    if params['cpu']:
        device = torch.device('cpu')
    else:
        device = torch.device(params['gpu_ids'][0])
    load_model(model, model_path, device)
    model.eval()
    src_data = torch.tensor([
        src_vocab.to_ints(src_text, params['data']['max_length'] + 1)
    ])
    src_data = src_data.to(device)
    print('Translating...this can take a minute...')
    out_data = beam_search(model, src_data, 1, params['data']['max_length'])
    out_text = tgt_vocab.to_text(out_data)[0]
    print(out_text)


def build_model(params, project_dir):
    """
    This builds a learner without loading data so that it loads more quickly.

    It is especially useful for doing predictions.
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
    if gpu_ids:
        device_id = gpu_ids[0]
        torch.cuda.set_device(device_id)
    else:
        device_id = None

    # Load vocabulary.
    vocab_path = os.path.join(params['data']['dir'], params['data']['vocab'])
    if os.path.isfile(vocab_path):
        # When using BERT, the source and target share a single vocabulary.
        src_vocab = VocabData(vocab_path)
        tgt_vocab = src_vocab
    else:
        src_vocab = VocabData(f'{vocab_path}.{params["data"]["src"]}')
        tgt_vocab = VocabData(f'{vocab_path}.{params["data"]["tgt"]}')

    # Define neural network.
    # Max length is 1 more than setting to account for BOS.
    bert_embedding_size = 768
    model = Pervasive(
        params['network']['block_sizes'],
        len(tgt_vocab),
        tgt_vocab.bos,
        params['data']['max_length'] + 1,
        params['data']['max_length'] + 1,
        bert_embedding_size,
        params['encoder']['embedding_dim'],
        params['encoder']['embedding_dropout'],
        params['network']['dropout'],
        params['decoder']['prediction_dropout'],
        params['network']['division_factor'],
        params['network']['growth_rate'],
        params['network']['bias'],
        params['network']['efficient'])

    model.init_weights()
    if device_id is not None:
        if not torch.cuda.is_available():
            raise ValueError(
                'Request to train on GPU {device_id}, but not GPU found.')
        model.cuda(device_id)

    return model, src_vocab, tgt_vocab


def load_model(model, filename, device):
    try:
        state = torch.load(filename, map_location=device)
        model.load_state_dict(state['model'], strict=True)
        print(f'Loaded model {filename}.')
    except FileNotFoundError:
        print(f'The model file {filename}.pth was not found!')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='translate French to English using a pervasive '
                    'attention model')
    parser.add_argument(
        'french',
        metavar='FRENCH_SENTENCE',
        help='french sentence to translate')
    parser.add_argument(
        '--config',
        help='configuration for model (default: config/fr2en.yaml)')
    parser.add_argument(
        '--model',
        help='path to model to load (default: data/fr2en/model.pth)')
    args = parser.parse_args()

    config = 'config/fr2en.yaml' if not args.config else args.config
    model = 'model/fr2en/model.pth' if not args.model else args.model

    translate(args.french, config, model)
