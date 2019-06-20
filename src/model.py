# Cf. https://github.com/dconathan/madpy-deploy-ml/blob/master/src/model.py

from typing import List, Tuple
import time
import functools
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# Set how many function calls to cache.
cache = functools.lru_cache(4096)


def translate_fren(fr_text: str) -> str:
    """
    """
    logger.debug(f'translating {len(fr_text)} chars of french, ' +
                 f'{len(fr_text.split())} tokens')
    t0 = time.time()

    model = get_model()
    en_text = model[fr_text] if fr_text in model else 'Unknown.'

    logger.debug(f'translating finished in {time.time() - t0:.2f}s')

    return en_text


def train():
    """
    Trains the model.
    """
    logger.info('training...')
    t0 = time.time()

    fr_texts, en_texts = get_data()

    logger.info(f'finished training in {time.time() - t0:.2f}s')


@cache
def get_model() -> dict:
    """
    Returns the Model object after caching it.
    """
    logger.info('getting and caching model')
    t0 = time.time()
    model = {
        'Réchauffez-vous!': 'Warm up!',
        'Bonjour le monde!': 'Hello world!'
    }
    logger.info(f'got model in {time.time() - t0:.2f}s')
    return model


def get_data() -> Tuple[List[str], List[int]]:
    """
    Returns the data as a list of sentences (strings) and their translations.
    """

    fr_texts = ['Bonjour le monde!']
    en_texts = ['Hello world!']
    return fr_texts, en_texts
