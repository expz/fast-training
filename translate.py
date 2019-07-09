import argparse
import os
from subprocess import Popen, PIPE
import torch

from config import parse_config
from evaluate import beam_search
from model import build_model, load_model
from pervasive import PervasiveOriginal
from vocab import VocabData


MAX_LENGTH = 120


def translate(src_text, config, model_path, beam=5):
    """
    Translate from a source language to a target language using
    the model at `model_path` whose config is described by the config
    at `config`.

    The translation uses a beam search of width `beam`.
    """
    params, project_dir = \
        parse_config(config, batch_size=1)

    # Tokenize the sentence.
    p = Popen([
        'perl', 'data/moses/tokenizer/tokenizer.perl', '-threads', '8', 
        '-a', '-l', 'fr'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate(src_text.encode('utf-8'))
    stdout = stdout.decode('utf-8')

    # Build PyTorch model.
    model, src_vocab, tgt_vocab = build_model(params, project_dir)

    # Load saved model.
    if params['cpu']:
        device = torch.device('cpu')
    else:
        device = torch.device(params['gpu_ids'][0])
    load_model(model, model_path, device)

    # Prepare input vector.
    src_toks = src_vocab.to_ints(stdout)[:MAX_LENGTH]
    src_data = torch.tensor([src_toks]).to(device)
    max_tgt_length = min(
        MAX_LENGTH, int(max(len(src_toks) * 1.5, len(src_toks) + 3)))

    # Beam search.
    out_data = beam_search(model, src_data, beam, max_tgt_length)
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
    model = 'model/fr2en.pth' if not args.model else args.model

    translate(args.french, config, model, args.beam)
