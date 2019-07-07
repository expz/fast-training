"""
Based on
  https://github.com/dconathan/madpy-deploy-ml/blob/master/src/model.py
"""

import functools
import io
import logging
import os
from subprocess import Popen, PIPE
import time
import torch
from typing import List, Tuple

from const import CONFIG_FILE, MODEL_FILE, TOKENIZER, MAX_LENGTH, BEAM_WIDTH
from config import parse_config
from evaluate import beam_search
from pervasive import PervasiveOriginal
from vocab import VocabData


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# Set how many function calls to cache.
cache = functools.lru_cache(4096)


def translate_fren(fr_text):
    """
    Translates a French sentence to English.
    """
    fr_text = fr_text[:1000].replace('\n', ' ')

    # Tokenize the sentence.
    p = Popen([
        'perl', 'data/moses/tokenizer/tokenizer.perl', '-threads', '8', 
        '-a', '-l', 'fr'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate(fr_text.encode('utf-8'))
    stdout = stdout.decode('utf-8')

    # Time the model execution.
    logger.debug(f'translating {len(fr_text)} chars of french, ' +
                 f'{len(stdout.split())} tokens')
    t0 = time.time()

    # Get the model and vocab (they are cached).
    model, src_vocab, tgt_vocab = get_model()

    # Prepare input vector.
    src_toks = src_vocab.to_ints(stdout)[:MAX_LENGTH]
    src_data = torch.tensor([src_toks])
    max_tgt_length = min(
        MAX_LENGTH, int(max(len(src_toks) * 1.5, len(src_toks) + 3)))

    # Beam search with width BEAM_WIDTH.
    out_data = beam_search(model, src_data, BEAM_WIDTH, max_tgt_length)
    en_text = tgt_vocab.to_text(out_data)[0]

    logger.debug(f'translating finished in {time.time() - t0:.2f}s')
    logger.debug(f"""
fr: {fr_text}
en: {en_text}
""")

    return en_text


def load_model(model, filename, device):
    try:
        state = torch.load(filename, map_location=device)
        model.load_state_dict(state['model'], strict=True)
        print(f'Loaded model {filename}.')
    except FileNotFoundError:
        raise Exception(f'The model file {filename} was not found!')


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
    src_vocab = VocabData(vocab_path)
    tgt_vocab = src_vocab

    # Define neural network.
    # Max length is 1 more than setting to account for BOS.
    model = PervasiveOriginal(
        params['network']['block_sizes'],
        len(tgt_vocab),
        tgt_vocab.bos,
        params['data']['max_length'] + 1,
        params['data']['max_length'] + 1,
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


@cache
def get_model():
    """
    Returns the Model object after caching it.
    """
    logger.info('getting and caching model')
    t0 = time.time()

    # Build model.
    params, project_dir = parse_config(CONFIG_FILE, batch_size=1)
    params['cpu'] = True
    model, src_vocab, tgt_vocab = build_model(params, project_dir)

    # Load saved model weights.
    device = torch.device('cpu')
    load_model(model, MODEL_FILE, device)

    logger.info(f'got model in {time.time() - t0:.2f}s')
    return model, src_vocab, tgt_vocab
