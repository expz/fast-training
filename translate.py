import argparse
import os
import sys
import torch

from config import parse_config
from evaluate import beam_search
from model import build_model, load_model
from pervasive import PervasiveOriginal
from vocab import VocabData


def translate(src_text, config, model_path, beam=5):
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
    out_data = beam_search(model, src_data, beam, params['data']['max_length'])
    out_text = tgt_vocab.to_text(out_data)[0]
    print(out_text)


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
    parser.add_argument(
        '--beam',
        default=5,
        type=int,
        help='width of beam for beam search (default: 5)')
    args = parser.parse_args()

    config = 'config/fr2en.yaml' if not args.config else args.config
    model = 'model/fr2en/model.pth' if not args.model else args.model

    translate(args.french, config, model, args.beam)
