import dask.array
import h5py
import logging
import math
import numpy as np
import os
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel
import random
import subprocess
import torch
import urllib


normalize_punct = (
    'https://raw.githubusercontent.com/moses-smt/mosesdecoder/master'
    '/scripts/tokenizer/normalize-punctuation.perl')
remove_nonprint = (
    'https://raw.githubusercontent.com/moses-smt/mosesdecoder/master'
    '/scripts/tokenizer/remove-non-printing-char.perl')
tokenizer_url = (
    'https://raw.githubusercontent.com/moses-smt/mosesdecoder/master'
    '/scripts/tokenizer/tokenizer.perl')

logger = logging.getLogger('fr2en')


class LanguageCorpus:
    """
    This is the most basic corpus and base class for other corpora.

    It uses a perl script from Moses to tokenize and `subword-nmt` to form
    BPE vocabulary. It outputs sequences of integers indexing into the
    vocabulary.

    Moses is available at https://github.com/moses-smt/mosesdecoder.
    """

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    def __init__(self,
                 name,
                 shuffle=True,
                 max_length=200):
        """`max_length` is the maximum length of a sentence in BPE tokens."""
        self.name = name
        self.shuffle = shuffle
        self.max_length = max_length

    def _clean(self, datafiles, max_size=None, use_cache=False):
        """
        Downloads Moses perl scripts if necessary, and uses them to normalize
        punctuation and remove non-printable characters.
        """
        # Download datafiles.
        normpunct_fn = normalize_punct.split('/')[-1]
        normpunct_path = os.path.join(self.data_dir, normpunct_fn)
        remnon_fn = remove_nonprint.split('/')[-1]
        remnon_path = os.path.join(self.data_dir, remnon_fn)
        if not os.path.isfile(normpunct_path):
            urllib.request.urlretrieve(normalize_punct, filename=normpunct_path)
        if not os.path.isfile(remnon_path):
            urllib.request.urlretrieve(remove_nonprint, filename=remnon_path)

        # Prepare an output directory.
        out_path = os.path.join(self.data_dir, self.name, 'train.tok')
        os.makedirs(os.path.join(self.data_dir, self.name), exist_ok=True)

        # Concatenate datasets for each language.
        langs = set()
        for dataset in datafiles:
            for lang in dataset:
                langs.add(lang)
                if not use_cache or not os.path.isfile(f'{out_path}.{lang}'):
                    os.system(f'cat {dataset[lang]} >> tmp.{lang}')

        # Clean datasets for each language.
        for lang in langs:
            if not use_cache or not os.path.isfile(f'{out_path}.{lang}'):
                logger.info(f'Cleaning {lang} dataset {dataset[lang]}.')
                max_size = 100000000000 if max_size is None else max_size
                os.system(f'head -n {max_size} tmp.{lang} '
                          f'| perl {normpunct_path} {lang} '
                          f'| perl {remnon_path} > {out_path}.{lang}')
                os.system(f'rm -rf tmp.{lang}')
            else:
                logger.info(
                    f'Using previously cleaned dataset {out_path}.{lang}.')

        return out_path, list(langs)

    def _tokenize(self, data_path, langs):
        """Tokenizes into BPE tokens using a perl script from Moses."""
        tokenizer_fn = remove_nonprint.split('/')[-1]
        tokenizer_path = os.path.join(self.data_dir, tokenizer_fn)
        if not os.path.isfile(tokenizer_path):
            urllib.request.urlretrieve(tokenizer_url, filename=tokenizer_path)
        tok_path = os.path.join(self.data_dir, self.name, 'tokens')
        for lang in langs:
            if not os.path.isfile(f'{tok_path}.{lang}'):
                logger.info(f'Tokenizing dataset {data_path}.{lang}.')
                os.system(
                    f'cat {data_path}.{lang} '
                    f'| perl {tokenizer_path} -threads 8 -a -l {lang} '
                    f'> {tok_path}.{lang}')
            else:
                logger.info(
                    f'Using previously tokenized dataset {data_path}.{lang}')
        return tok_path

    def _encode(self, tok_path, langs, joint_vocab_size):
        """
        Tokenizes sentences using `subword-nmt` and converts them to sequences
        of integers.
        """
        vocab_path = os.path.join(self.data_dir, self.name, 'vocab')
        codes_path = os.path.join(self.data_dir, self.name, 'bpe_codes')
        learn_cmd = (
            'subword-nmt learn-joint-bpe-and-vocab '
            f'--input {tok_path}.{langs[0]} {tok_path}.{langs[1]}'
            f'-s {joint_vocab_size} -o {codes_path} --write-vocabulary'
            f'{vocab_path}.{langs[0]} {vocab_path}.{langs[1]}')
        os.system(learn_cmd)
        bpe_toks = {}
        for lang in langs:
            with open(f'{vocab_path}.{lang}', 'r') as f_vocab:
                vocab = f_vocab.read().split('\n')
            vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + vocab
            with open(f'{vocab_path}.{lang}', 'w') as f_vocab:
                f_vocab.write('\n'.join(vocab))
            wtoi = {word: idx for idx, word in enumerate(vocab)}
            with open(f'{tok_path}.{lang}', 'r') as f_in:
                apply_cmd = [
                    'subword-nmt', 'apply-bpe', '-c', codes_path,
                    '--vocabulary', f'{vocab_path}.{lang}',
                    '--vocabulary-threshold', '50'
                ]
                bpe_sents = subprocess.check_output(
                    apply_cmd, stdin=f_in).decode('utf-8').split('\n')
            bpe_toks[lang] = [
                [wtoi[word] for word in sent.split()] for sent in bpe_sents
            ]
        return bpe_toks

    def _save(self, data, valid_size, dtype='int32'):
        """Saves the datasets to HDF5 files."""
        h5path = os.path.join(self.data_dir, self.name)
        for lang in data:
            with h5py.File(f'{h5path}/{lang}.h5', 'w') as f:
                train_ds = f.create_dataset(
                    'train', data=data[lang][:-valid_size])
                train_ds.attrs['dtype'] = dtype
                valid_ds = f.create_dataset(
                    'valid', data=data[lang][-valid_size:])
                valid_ds.attrs['dtype'] = dtype
        return [f'{h5path}/{lang}.h5' for lang in data]

    def _shuffle(self, toks):
        """Shuffles the sentences in `toks`."""
        logging.info('Shuffling datasets.')
        new_toks = {}
        toks_list = list(zip(*[toks[lang] for lang in toks]))
        random.shuffle(toks_list)
        d = list(zip(*toks_list))
        for i, lang in enumerate(toks):
            new_toks[lang] = d[i]
        return new_toks

    def create(self, datafiles, joint_vocab_size, max_size=None, valid_size=0,
               use_cache=False):
        """Creates train and validation datasets from files `datafiles`."""
        out_path, langs = self._clean(datafiles, max_size, use_cache)
        tok_path = self._tokenize(out_path, langs)
        bpe_toks = self._encode(tok_path, langs, joint_vocab_size)
        if self.shuffle:
            bpe_toks = self._shuffle(bpe_toks)
        return self._save(bpe_toks, valid_size, dtype='int32')


class BertCorpus(LanguageCorpus):
    """
    This is a `LanguageCorpus` which uses BERT's multilingual BPE vocabulary
    to tokenize.

    BERT's multilingual vocabulary supports 100 languages in one, so it has
    approximately 114,000 tokens.
    """
    def __init__(self,
                 name,
                 shuffle=True,
                 max_length=200):
        super().__init__(name, shuffle, max_length)
        # These are tokens '[CLS]', '[SEP]', '[PAD]'
        self.bos, self.eos, self.pad = 101, 102, 0
        self.emb_size = 768

    def _encode(self, out_path, langs):
        """
        Encodes sentences listed one per line in file `out_path` as sequences
        of integers indexing into the BERT multilingual vocabulary.
        """
        # Load saved tokenized data if we cached it during a previous run.
        toks_path = f'{out_path}.indices.pickle'
        if os.path.isfile(toks_path):
            logging.info(f'Loading BPE tokenized data from {toks_path}.')
            try:
                with open(toks_path, 'rb') as f:
                    toks, lengths = pickle.load(f)
                cnt = sum(1 for line in open(f'{out_path}.{langs[0]}', 'r'))
                if cnt == len(toks[langs[0]]):
                    return toks, lengths
                else:
                    logging.info(
                        'Cache file of BPE tokenized data is out of date. '
                        'Remaking.')
            except Exception as e:
                logging.warning(
                    f'Loading cached BPE tokenized data failed: {str(e)}.')

        # Load Bert tokenizer.
        logging.info(f'Encoding data as BPE token indices.')

        # WARNING: If you change the tokenizer, then make sure the above
        #          hard-coded bos, eos and pad token indices are correct.
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-cased', do_lower_case=False)

        # Tokenize the sentences in the given files.
        toks = {}
        lengths = {}
        empty_lines = set()
        for lang in langs:
            with open(f'{out_path}.{lang}', 'r') as f:
                logging.info(f'Converting {lang} text to BPE token indices.')
                ts = [
                    tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(sent))[:self.max_length]
                    for sent in f
                ]
                lengths[lang] = [len(sent) for sent in ts]
                empty_lines.update(
                    [i for i, ll in enumerate(lengths[lang]) if ll == 0])

                # Vectors will have length `max_len + 1` to account for BOS.
                max_len = max(lengths[lang])

                logging.info(f'Adding BOS, EOS and PAD tokens for {lang}.')
                toks[lang] = [
                    ([self.bos] + sent + [self.eos]
                        + [self.pad] * (max_len - len(sent) - 1))[:max_len + 1]
                    for sent in ts
                ]

        # Remove pairs of sentences with at least one empty sentence.
        for lang in langs:
            toks[lang] = [
                sent
                for i, sent in enumerate(toks[lang]) if i not in empty_lines
            ]
            lengths[lang] = [
                l for i, l in enumerate(lengths[lang]) if i not in empty_lines
            ]

        # Save vocabulary to file. (It will be called `vocab.txt`.)
        vocab_dir = os.path.join(self.data_dir, self.name)
        tokenizer.save_vocabulary(vocab_dir)

        # Save BPE tokenized data so we do not have to recompute if we rerun.
        with open(toks_path, 'wb') as f:
            logging.info(f'Saving BPE tokenized data to {toks_path}.')
            pickle.dump((toks, lengths), f)
        return toks, lengths

    def _save_with_lens(self, data, lens, valid_size, dtype='int32'):
        """
        Saves the datasets to one HDF5 file per language together with
        the list of the sentence lengths.

        This separates `valid_size` sentences from the end of the training
        dataset to form the validation set.
        """
        h5path = os.path.join(self.data_dir, self.name)
        for lang in data:
            with h5py.File(f'{h5path}/{lang}.h5', 'w') as f:
                train_ds = f.create_dataset(
                    'train', data=data[lang][:-valid_size])
                train_ds.attrs['dtype'] = dtype
                train_lens_ds = f.create_dataset(
                    'train_lens', data=lens[lang][:-valid_size])
                train_lens_ds.attrs['dtype'] = dtype
                valid_ds = f.create_dataset(
                    'valid', data=data[lang][-valid_size:])
                valid_ds.attrs['dtype'] = dtype
                valid_lens_ds = f.create_dataset(
                    'valid_lens', data=lens[lang][-valid_size:])
                valid_lens_ds.attrs['dtype'] = dtype
        return [f'{h5path}/{lang}.h5' for lang in data]

    def _shuffle_with_lens(self, toks, lens):
        """Shuffles datasets which have associated sentence length lists."""
        logging.info('Shuffling datasets.')
        new_toks, new_lens = {}, {}
        toks_lens = (
            [toks[lang] for lang in toks] + [lens[lang] for lang in lens])
        toks_lens = list(zip(*toks_lens))
        random.shuffle(toks_lens)
        d = list(zip(*toks_lens))
        for i, lang in enumerate(toks):
            new_toks[lang] = d[i]
            new_lens[lang] = d[i + len(toks)]
        return new_toks, new_lens

    def create(self, datafiles, max_size=None, valid_size=0, use_cache=False):
        """Creates train and validation datasets from files `datafiles`."""
        out_path, langs = self._clean(datafiles, max_size, use_cache)
        toks, lens = self._encode(out_path, langs)
        if self.shuffle:
            toks, lens = self._shuffle_with_lens(toks, lens)
        return self._save_with_lens(toks, lens, valid_size, dtype='int32')


class LowResolutionCorpus(BertCorpus):

    def __init__(self,
                 name,
                 shuffle=True,
                 max_length=200):
        super().__init__(name, shuffle, max_length)

    def _subsample(self, toks, lens, p):
        """
        Creates a new dataset which contains each sentence from `tok` starting
        with BOS and the first token of the sentence and with `p` percent total
        tokens randomly kept. The rest of the tokens are discarded.

        The indices of discarded tokens agree across languages.
        """
        max_len = len(toks[toks.keys()[0]][0])
        n = math.floor(max_len * p)
        new_toks = {}
        new_lens = {}
        for lang in toks:
            new_toks[lang] = []
            new_lens[lang] = []
            for sent in toks[lang]:
                indices = list(range(1, max_len))
                random.shuffle(indices)
                indices = sorted(indices[:n])
                # TODO: Fix me.
                new_toks[lang].append([sent[i] for i in indices])
        return new_toks, new_lens

    def create(self, datafiles, p=0.5, valid_size=0, shuffle=True,
               max_size=None, use_cache=False):
        """
        Create the dataset from `datafiles` by keeping `p` percent of
        the input/output tokens.
        """
        out_path, langs = self._clean(datafiles, max_size, use_cache)
        toks, lens = self._encode(out_path, langs)
        if self.shuffle:
            for lang in toks:
                toks_lens = list(zip(toks[lang], lens[lang]))
                random.shuffle(toks_lens)
                toks[lang], lens[lang] = zip(*toks_lens)
        toks, lens = self._subsample(toks, lens)
        return self._save_with_lens(toks, lens, valid_size, dtype='int32')


class WindowedCorpus(BertCorpus):
    """
    This is a corpus formed by selecting a window of tokens of length
    `window_size` from another corpus.

    The window is applied at two positions.

    1. At the beginning of the sentence. This ensures the model learns how
       to begin a sentence.
    2. Starting at the middle of the sentence. This requires knowledge of the
       length of each sentence.
    """

    def __init__(self,
                 name,
                 shuffle=True,
                 max_length=200):
        super().__init__(name, shuffle, max_length)

    def _window(self, toks, lens, window_size):
        """
        Selects two windows of size `window_size` from each sentence in
        `toks` that has length given in `lens`.
        """
        # lens do not include BOS or EOS tokens.
        new_toks = {}
        new_lens = {}
        for lang in toks:
            new_toks[lang] = []
            new_lens[lang] = []
            for i, sent in enumerate(toks[lang]):
                n = lens[lang][i]
                new_toks[lang].append(sent[:window_size])
                new_lens[lang].append(min(n, window_size - 1))
                new_toks[lang].append(sent[n // 2:n // 2 + window_size])
                new_lens[lang].append(min(n - n // 2, n // 2 + window_size))
        return new_toks, new_lens

    def create(self, datafiles, max_size=None, window_size=25, valid_size=0,
               use_cache=False):
        """
        Create a dataset from `datafiles` by randomly selecting a window
        of `window_size` tokens from every sentence.
        """
        out_path, langs = self._clean(datafiles, max_size, use_cache)
        toks, lens = self._encode(out_path, langs)
        if self.shuffle:
            toks, lens = self._shuffle_with_lens(toks, lens)
        toks, lens = self._window(toks, lens)
        return self._save_with_lens(toks, lens, valid_size, dtype='int32')


class EmbeddingCorpus(BertCorpus):
    """
    This class represents a corpus composed of embedding vectors.
    """

    def __init__(self,
                 name,
                 shuffle=True,
                 max_length=200):
        super().__init__(name, shuffle, max_length)
        self.bos_emb, self.eos_emb, self.pad_emb = None, None, None

    def _embed(self, toks):
        """
        This converts the lists of integers in `toks` are converted to
        embedding vectors using BERT's multlingual case model.
        """

        def apply_emb(x):
            """
            This function applies the BERT embedding layer to `x`. It is
            called by the mapping function. It must be a sub-function so
            it has access to `bert_emb`.

            `dask.array.map_blocks()` requires the mapping function to
            always return an array with the same shape as the calling array's
            `chunksize`.
            """
            emb = np.array(bert_emb(torch.LongTensor(x)), dtype=np.int32)
            if x.shape[0] < chunk_size:
                dims = (chunk_size - x.shape[0], max_length, self.emb_size)
                return np.concatenate((emb, np.zeros(dims, dtype=np.int32)))
            return emb

        bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        bert_emb = bert_model.embeddings.word_embeddings
        embs = {}
        chunk_size = 1024
        for lang in toks:
            max_length = len(toks[lang][0])
            toks[lang] = dask.array.from_array(
                np.array(toks[lang], dtype=np.int32),
                chunks=(chunk_size, max_length))
            logger.info(f'Calculating embeddings for language {lang}.')
            embs[lang] = toks[lang].map_blocks(
                apply_emb,
                chunks=(chunk_size, max_length, self.emb_size),
                dtype=np.float32,
                new_axis=[2])
        self.bos_emb = \
            np.array(bert_emb(torch.tensor([self.bos]))[0], dtype=np.float32)
        self.eos_emb = \
            np.array(bert_emb(torch.tensor([self.eos]))[0], dtype=np.float32)
        self.pad_emb = \
            np.array(bert_emb(torch.tensor([self.pad]))[0], dtype=np.float32)
        return embs

    def _save(self, embs, valid_size):
        """Saves the dask arrays in `embs` to HDF5 files."""
        h5path = os.path.join(self.data_dir, self.name)
        h5files = []
        for lang in embs:
            h5file = f'{h5path}/{lang}.h5'
            h5files.append(h5file)
            embs[lang][:-valid_size].to_hdf5(h5file, 'train')
            embs[lang][-valid_size:].to_hdf5(h5file, 'valid')
            with h5py.File(h5file, 'w') as f:
                f['train'].attrs['dtype'] = 'float32'
                f['valid'].attrs['dtype'] = 'float32'
        return h5files

    def create(self, datafiles, max_size=None, valid_size=0, use_cache=False):
        """Creates train and validation datasets from files `datafiles`."""
        out_path, langs = self._clean(datafiles, max_size, use_cache)
        toks, _ = self._encode(out_path, langs)
        if self.shuffle:
            toks = self._shuffle(toks)
        embs = self._embed(toks)

        # Save the datasets to an hdf5 file on disk.
        return self._save(embs, valid_size)


class LowResolutionEmbeddingCorpus(EmbeddingCorpus):
    """
    This is a corpus of BERT embedding vectors which have been averaged
    by a sliding window of size `window_size` moving `window_step` tokens
    each step.

    The EOS and PAD tokens are preserved *without* averaging.
    """

    def __init__(self,
                 name,
                 window_step=2,
                 window_size=2,
                 shuffle=True,
                 max_length=200):
        super().__init__(name, shuffle, max_length)
        self.window_step = window_step
        self.window_size = window_size

    def _avg_embs(self, embs, lengths):
        """
        Averages the embeddings of `embs` which represent sentences with
        lengths given by `lengths`.
        """

        def eos_and_pad(emb):
            """
            Restore EOS marker and PAD tokens after it. This is called by
            `apply_along_axis()`.

            This must be a subfunction of `_avg_embs()` so that it has access
            to the `max_length` variable.
            """
            n = int(round(emb[0]))
            row = n // max_len
            col = n % max_len
            if row >= len(lengths):
                return emb
            elif lengths[row] == col - 1:
                return eos_emb
            elif (col - 1 > lengths[row]
                    and col <= lengths[row] + self.window_size):
                return pad_emb
            return emb

        logger.info('Calcuating average embeddings.')
        bos = (self.bos_emb.reshape((1, 1, self.emb_size))
                           .repeat(embs.shape[0], axis=0))
        avg_embs = dask.array.concatenate(
            [bos] + [
                embs[:, i:i + self.window_size, :].mean(
                    axis=1, keepdims=True)
                for i in range(1, embs.shape[1], self.window_step)
            ], axis=1).astype(np.float32)

        # Add a coordinate to the front of every embedding vector containing
        # a number that determines the sentence and token of the vector.
        # This is the only way to get that info to `eos_and_pad`.
        eos_emb = np.concatenate([[-1], self.eos_emb])
        pad_emb = np.concatenate([[-1], self.pad_emb])
        max_len = int(avg_embs.shape[1])
        indices = dask.array.arange(avg_embs.shape[0] * max_len)
        indices = indices.reshape((avg_embs.shape[0], max_len, 1))
        avg_embs = dask.array.concatenate([indices, avg_embs], axis=2)
        avg_embs = avg_embs.rechunk((1024, max_len, len(eos_emb)))

        # The dask version of `apply_along_axis()` is broken or does not behave
        # like the numpy version, so we have to use `map_blocks()`.
        logger.info('Fixing EOS and PAD tokens.')
        avg_embs = avg_embs.map_blocks(
            lambda b: np.apply_along_axis(eos_and_pad, 2, b),
            chunks=(1024, max_len, len(eos_emb)),
            dtype=np.float32)

        # Drop indices that were stored as first embedding dimension.
        avg_embs = avg_embs[:, :, 1:]
        return avg_embs

    def _save(self, avg_embs, valid_size):
        """
        Saves the dask arrays containing averaged embeddings to HDF5 files.
        """
        h5path = os.path.join(self.data_dir, self.name)
        h5files = []
        for lang in avg_embs:
            logger.info(f'Saving HDF5 file for language {lang}.')
            h5file = f'{h5path}/{lang}.h5'
            h5files.append(h5file)
            print(h5file)
            with h5py.File(h5file, 'w') as f_out:
                # Use `store()`. `to_hdf5` produces empty fie for some reason.
                train = avg_embs[lang][:-valid_size]
                train_ds = f_out.require_dataset(
                    'train', shape=train.shape, dtype=train.dtype)
                train_ds.attrs['dtype'] = 'float32'
                dask.array.store(train, train_ds)
                valid = avg_embs[lang][-valid_size:]
                valid_ds = f_out.require_dataset(
                    'valid', shape=valid.shape, dtype=valid.dtype)
                valid_ds.attrs['dtype'] = 'float32'
                dask.array.store(valid, valid_ds)

    def create(self, datafiles, max_size=None, valid_size=0, use_cache=False):
        """Creates train and validation datasets from files `datafiles`."""
        out_path, langs = self._clean(datafiles, max_size, use_cache)
        toks, lens = self._encode(out_path, langs)
        if self.shuffle:
            toks, lens = self._shuffle_with_lens(toks, lens)
        embs = self._embed(toks)
        avg_embs = {}
        for lang in langs:
            avg_embs[lang] = self._avg_embs(embs[lang], lens[lang])
        return self._save(avg_embs, valid_size)
