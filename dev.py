#!/usr/bin/env python
"""
This script launches model training in separate processes, one for each GPU.
"""

import fire
import logging
import multiprocessing
import os
import streamlit
import sys
import tarfile
import torch

from config import parse_config
from corpus import (
    LanguageCorpus, BertCorpus, EmbeddingCorpus,
    LowResolutionEmbeddingCorpus)
from evaluate import beam_search
from train import build_learner, train_worker


logger = logging.getLogger('fr2en')


def extract(fn, output_dir):
    """
    Extract a .tar, .tgz or .gz file to `output_dir`.
    """
    if fn.endswith('.tar.gz') or fn.endswith('.tgz'):
        with tarfile.open(fn, 'r:gz') as f:
            f.extractall(output_dir)
    elif fn.endswith('.tar'):
        with tarfile.open(fn, 'r') as f:
            f.extractall(output_dir)


class PrepareData:
    """
    This class encapsulates the commands for preparing various datasets.
    """

    _urls = {
        'ca-parliament-house':
            'http://www.isi.edu/natural-language/download/hansard/'
            'hansard.36.r2001-1a.house.debates.training.tar',
        'ca-parliament-senate':
            'http://www.isi.edu/natural-language/download/hansard/'
            'hansard.36.r2001-1a.senate.debates.training.tar',
        'commoncrawl':
            'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
        'europarl-de-en':
            'http://www.statmt.org/europarl/v7/de-en.tgz',
        'europarl-es-en':
            'http://www.statmt.org/europarl/v7/es-en.tgz',
        'europarl-it-en':
            'http://www.statmt.org/europarl/v7/it-en.tgz',
        'europarl-fr-en':
            'http://www.statmt.org/europarl/v7/fr-en.tgz',
        'europarl-sl-en':
            'http://www.statmt.org/europarl/v7/sl-en.tgz',
        'news2014':
            'http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz',
        'news2016-de-en':
            'http://www.casmacat.eu/corpus/news-commentary/'
            'news-commentary-v11.de-en.xliff.gz',
        'news2016-fr-en':
            'http://www.casmacat.eu/corpus/news-commentary/'
            'news-commentary-v11.fr-en.xliff.gz',
    }

    _corpora = {
        'ca-parliament-house-fr-en': {
            'en': 'hansard.36/Release-2001.1a/sentence-pairs/house/debates/'
                  'development/training/hansard.36.1.house.debates',
            'fr': 'hansard.36/Release-2001.1a/sentence-pairs/house/debates/'
                  'development/training/hansard.36.1.house.debates'
        },
        'ca-parliament-senate-fr-en': {
            'en': 'hansard.36/Release-2001.1a/sentence-pairs/senate/debates/'
                  'development/training/hansard.36.1.senate.debates',
            'fr': 'hansard.36/Release-2001.1a/sentence-pairs/senate/debates/'
                  'development/training/hansard.36.1.senate.debates'
        },
        'commoncrawl-fr-en': {
            'en': 'commoncrawl.fr-en.en',
            'fr': 'commoncrawl.fr-en.fr'
        },
        'europarl-de-en': {
            'en': 'europarl-v7.de-en.en',
            'de': 'europarl-v7.de-en.de'
        },
        'europarl-es-en': {
            'en': 'europarl-v7.es-en.en',
            'es': 'europarl-v7.es-en.es'
        },
        'europarl-it-en': {
            'en': 'europarl-v7.it-en.en',
            'it': 'europarl-v7.it-en.it'
        },
        'europarl-fr-en': {
            'en': 'europarl-v7.fr-en.en',
            'fr': 'europarl-v7.fr-en.fr'
        },
        'europarl-sl-en': {
            'en': 'europarl-v7.sl-en.en',
            'sl': 'europarl-v7.sl-en.sl'
        },
        'news2014-fr-en': {
            'en': 'training/news-commentary-v9.fr-en.en',
            'fr': 'training/news-commentary-v9.fr-en.fr'
        },
        'news2016-fr-en': {
            'en': 'news-commentary-v11.fr-en.xliff',
            'fr': 'news-commentary-v11.fr-en.xliff'
        },
    }

    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _tmp_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'downloads', 'tmp')

    def _download(self, data):
        """
        Downloads the datasets listed in `data`. They should be strings which
        are keys of the `self._urls` dictionary.

        Datasets are downloaded to a directory common to all models. If they
        already exist, they are not re-downloaded.

        Extracts datasets as well.
        """
        os.makedirs(self._tmp_dir, exist_ok=True)
        for dataset in data:
            if dataset not in self._urls:
                raise ValueError(f'Unknown dataset: {dataset}')
            out_fn = os.path.join(
                self._data_dir, 'downloads', self._urls[dataset].split('/')[-1])
            if not os.path.isfile(out_fn):
                logger.info(f'Downloading dataset {self._urls[dataset]}.')
                os.system(f'wget -O {out_fn} {self._urls[dataset]}')
            logger.info(f'Extracting dataset {out_fn}.')
            extract(out_fn, self._tmp_dir)

    def _datafiles(self, lang, data):
        """
        Returns a list of dictionaries which have the dataset file paths
        for each language.
        """
        # TODO: Implement special handling of Canadian Parliament proceedings.
        #       The zip files extract to hundreds of small *.gz files.
        datafiles = []
        for dataset in data:
            if 'ca-parliament' in dataset:
                raise NotImplementedError(
                    'Canadian Parliament datasets are not yet supported.')
            if dataset[-3:] != '-en':
                dataset = f'{dataset}-{lang}-en'
            datafiles.append({
                'en':
                    os.path.join(self._tmp_dir, self._corpora[dataset]['en']),
                lang:
                    os.path.join(self._tmp_dir, self._corpora[dataset][lang])
            })
        return datafiles

    def list_datasets(self):
        """Lists the available datasets."""
        print('\n'.join(self._urls.keys()))

    def standard(self, lang='fr', name=None, data=['news2014'], max_length=200,
                 max_size=None, shuffle=True, joint_vocab_size=40000,
                 valid_size=0, use_cache=False):
        """
        Creates a dataset of sequences of indices into a joint BPE vocabulary
        generated by `subword-nmt`.
        """
        name = name if name else f'standard-{lang}-en'
        self._download(data)
        datafiles = self._datafiles(lang, data)
        ds = LanguageCorpus(name, shuffle=shuffle, max_length=max_length)
        ds.create(datafiles, joint_vocab_size, max_size=max_size,
                  valid_size=valid_size, use_cache=use_cache)

    def bert(self, lang='fr', name=None, data=['news2014'], max_length=200,
             max_size=None, shuffle=True, valid_size=0, use_cache=False):
        """
        Creates a dataset of sequences of indices into the 100-language
        multilingual, cased BERT BPE vocabulary.
        """
        name = name if name else f'bert-{lang}-en'
        self._download(data)
        datafiles = self._datafiles(lang, data)
        ds = BertCorpus(name, shuffle=shuffle, max_length=max_length)
        ds.create(datafiles, max_size, valid_size, use_cache)

    def embed(self, lang='fr', name=None, data=['news2014'], max_length=200,
              max_size=None, shuffle=True, valid_size=0, use_cache=False):
        """
        Creates a dataset of BERT embeddings of sentence tokens.
        """
        name = name if name else f'embed-{lang}-en'
        self._download(data)
        datafiles = self._datafiles(lang, data)
        ds = EmbeddingCorpus(name, shuffle=shuffle, max_length=max_length)
        ds.create(datafiles, max_size, valid_size, use_cache)

    def low_res_embed(self, step, size, lang='fr', name=None, data=['news2014'],
                      max_length=200, max_size=None, shuffle=True,
                      valid_size=0, use_cache=False):
        """
        Creates a dataset of BERT embeddings averaged using a window of size
        `size` moving `step` tokens per step.
        """
        name = name if name else f'embed-{lang}-en'
        self._download(data)
        datafiles = self._datafiles(lang, data)
        ds = LowResolutionEmbeddingCorpus(
            name, step, size, shuffle=shuffle, max_length=max_length)
        ds.create(datafiles, max_size, valid_size, use_cache)


class PervasiveApp(object):
    """
    This is the command line app that the `fire` packages exposes
    using command line arguments.
    """

    def __init__(self):
        self.prepare_data = PrepareData()

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
        params, project_dir = parse_config(config, device_ids, lr, batch,
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
        if nprocs > 1:
            logger.info("""

==================================================================

Training on multiple GPUs may cause spurious error messages at the
beginning even though the training works fine!

==================================================================

""")
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
        params, project_dir = parse_config(config, [gpu_id], batch_size=batch)
        learn, src_vocab, tgt_vocab = build_learner(params, project_dir)
        self._restore(learn, restore)
        batch, tgt = next(iter(learn.data.valid_dl))
        src_data, tgt_data = \
            batch.split([learn.model.Ts, learn.model.Tt], dim=1)
        src_text = src_vocab.to_text(src_data)
        out_data = beam_search(
            learn.model, src_data, 5, params['data']['max_length'])
        out_text = tgt_vocab.to_text(out_data)
        for src, out in zip(src_text, out_text):
            print(f'IN: {src}')
            print(f'OUT: {out}')

    def find_lr(self, config, gpu_id=0):
        """
        Search for an optimal learning rate and print plot to Streamlit.
        """
        if 'streamlit' not in sys.modules:
            print('Please install streamlit or dev-requirements.txt '
                  'to use this feature.')
            sys.exit(1)
        print('************************************************************')
        print('TESTING LEARNING RATES: THIS WILL RUN FOR UNDER 100 BATCHES.')
        print('************************************************************\n')
        params, project_dir = parse_config(config, [gpu_id])
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
