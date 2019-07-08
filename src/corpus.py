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


nonbreaking_url = (
    'https://raw.githubusercontent.com/moses-smt/mosesdecoder'
    '/ef028446f3640e007215b4576a4dc52a9c9de6db/scripts/share'
    '/nonbreaking_prefixes/nonbreaking_prefix')
normalize_punct_url = (
    'https://raw.githubusercontent.com/moses-smt/mosesdecoder'
    '/ef028446f3640e007215b4576a4dc52a9c9de6db/scripts/tokenizer'
    '/normalize-punctuation.perl')
remove_nonprint_url = (
    'https://raw.githubusercontent.com/moses-smt/mosesdecoder'
    '/ef028446f3640e007215b4576a4dc52a9c9de6db/scripts/tokenizer'
    '/remove-non-printing-char.perl')
tokenizer_url = (
    'https://raw.githubusercontent.com/moses-smt/mosesdecoder'
    '/ef028446f3640e007215b4576a4dc52a9c9de6db/scripts/tokenizer'
    '/tokenizer.perl')

logger = logging.getLogger('fr2en')


class LanguageCorpus:
    """
    This is the most basic corpus and base class for other corpora.

    It uses a perl script from Moses to tokenize and `subword-nmt` to form
    BPE vocabulary. These are standard tools for preprocessing, see e.g.
    https://github.com/pytorch/fairseq/blob/master/examples/translation
           /prepare-wmt14en2de.sh

    It outputs sequences of integers indexing into the vocabulary.

    Moses is available at https://github.com/moses-smt/mosesdecoder.
    """

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    moses_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'moses')

    def __init__(self,
                 name,
                 shuffle=True,
                 max_length=200):
        """`max_length` is the maximum length of a sentence in BPE tokens."""
        self.name = name
        self.shuffle = shuffle
        self.max_length = max_length
        os.makedirs(os.path.join(self.moses_dir, 'tokenizer'), exist_ok=True)

    def _clean(self, datafiles, max_size=None, use_cache=False):
        """
        Downloads Moses perl scripts if necessary, and uses them to normalize
        punctuation and remove non-printable characters.
        """
        # Download datafiles.
        normpunct_fn = normalize_punct_url.split('/')[-1]
        normpunct_path = os.path.join(self.moses_dir, 'tokenizer', normpunct_fn)
        remnon_fn = remove_nonprint_url.split('/')[-1]
        remnon_path = os.path.join(self.moses_dir, 'tokenizer', remnon_fn)
        if not os.path.isfile(normpunct_path):
            urllib.request.urlretrieve(
                normalize_punct_url, filename=normpunct_path)
        if not os.path.isfile(remnon_path):
            urllib.request.urlretrieve(
                remove_nonprint_url, filename=remnon_path)

        # Prepare an output directory.
        out_path = os.path.join(self.data_dir, self.name, 'cleaned')
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
                logger.info(f'Cleaning {lang} combined dataset.')
                max_size = 100000000000 if max_size is None else max_size
                os.system(f'head -n {max_size} tmp.{lang} '
                          f'| perl {normpunct_path} {lang} '
                          f'| perl {remnon_path} > {out_path}.{lang}')
                os.system(f'rm -rf tmp.{lang}')
            else:
                logger.info(
                    f'Using previously cleaned dataset {out_path}.{lang}.')

        return out_path, list(langs)

    def _tokenize(self, data_path, langs, use_cache=False):
        """Tokenizes into BPE tokens using a perl script from Moses."""
        tokenizer_fn = tokenizer_url.split('/')[-1]
        tokenizer_path = os.path.join(self.moses_dir, 'tokenizer', tokenizer_fn)
        if not os.path.isfile(tokenizer_path):
            urllib.request.urlretrieve(tokenizer_url, filename=tokenizer_path)
        nonbreaking_dir = \
            os.path.join(self.moses_dir, 'share', 'nonbreaking_prefixes')
        os.makedirs(nonbreaking_dir, exist_ok=True)
        nonbreaking_fn = nonbreaking_url.split('/')[-1]
        nonbreaking_path = os.path.join(nonbreaking_dir, nonbreaking_fn)
        for lang in langs:
            if not os.path.isfile(f'{nonbreaking_path}.{lang}'):
                urllib.request.urlretrieve(
                    f'{nonbreaking_url}.{lang}',
                    filename=f'{nonbreaking_path}.{lang}')
        tok_path = os.path.join(self.data_dir, self.name, 'tokens')
        for lang in langs:
            if not use_cache or not os.path.isfile(f'{tok_path}.{lang}'):
                logger.info(f'Tokenizing dataset {data_path}.{lang}.')
                os.system(
                    f'cat {data_path}.{lang} '
                    f'| perl {tokenizer_path} -threads 8 -a -l {lang} '
                    f'> {tok_path}.{lang}')
            else:
                logger.info(
                    f'Using previously tokenized dataset {data_path}.{lang}')
        return tok_path

    def _filter_sents(self, tok_path, langs, use_cache=False):
        logging.info('Filtering out sentence pairs with invalid lengths.')

        # Filter out sentence pairs with invalid lengths.
        if (not use_cache
                or not os.path.isfile(f'{tok_path}.filtered.{langs[0]}')
                or not os.path.isfile(f'{tok_path}.filtered.{langs[1]}')):
            with open(f'{tok_path}.{langs[0]}', 'r') as f, \
                    open(f'{tok_path}.{langs[1]}', 'r') as g, \
                    open(f'{tok_path}.filtered.{langs[0]}', 'w') as f_out, \
                    open(f'{tok_path}.filtered.{langs[1]}', 'w') as g_out:
                line1 = f.readline()
                line2 = g.readline()
                while line1 and line2:
                    l1 = len(line1.split())
                    l2 = len(line2.split())
                    if ((not (l1 > 1.5 * l2 or l2 > 1.5 * l1)
                            or (l1 <= 10 and l2 <= 10)) and l1 > 0 and l2 > 0):
                        # readline() keeps the newline, write() does not add one
                        f_out.write(line1)
                        g_out.write(line2)
                    line1 = f.readline()
                    line2 = g.readline()

    def _encode(self, tok_path, langs, joint_vocab_size, use_cache=False):
        """
        Tokenizes sentences using `subword-nmt` and converts them to sequences
        of integers.
        """
        # Learn joint BPE.
        vocab_path = os.path.join(self.data_dir, self.name, 'vocab')
        freqs_path = os.path.join(self.data_dir, self.name, 'freqs')
        codes_path = os.path.join(self.data_dir, self.name, 'bpe_codes')
        bpe_path = os.path.join(self.data_dir, self.name, 'int_toks')
        if (not use_cache or not os.path.isfile(f'{freqs_path}.{langs[0]}')
                or not os.path.isfile(codes_path)):
            logging.info('Learning joint BPE.')
            learn_cmd = (
                'subword-nmt learn-joint-bpe-and-vocab '
                f'--input {tok_path}.{langs[0]} {tok_path}.{langs[1]} '
                f'-s {joint_vocab_size // 2} -o {codes_path} '
                f'--write-vocabulary '
                f'{freqs_path}.{langs[0]} {freqs_path}.{langs[1]}')
            os.system(learn_cmd)
        else:
            logging.info('Using previously learned joint BPE.')

        logging.info(f'Preparing joint vocabulary of size at most '
                     f'{joint_vocab_size + 4}.')

        self._filter_sents(tok_path, langs, use_cache)

        # Add special tokens to frequencies (word plus num of occurrences).
        freqs = ['[PAD] 1000', '[UNK] 1000', '[CLS] 1000', '[SEP] 1000']
        with open(f'{freqs_path}.{langs[0]}', 'r') as f_freqs, \
                open(f'{freqs_path}.{langs[1]}', 'r') as g_freqs:
            line1 = f_freqs.readline()
            line2 = g_freqs.readline()
            seen = set()
            while line1 and line2:
                f1 = line1.split()
                f2 = line2.split()
                while len(f1) < 2 or f1[0] in seen:
                    line1 = f_freqs.readline()
                    f1 = line1.split()
                seen.add(f1[0])
                while len(f2) < 2 or f2[0] in seen:
                    line2 = g_freqs.readline()
                    f2 = line2.split()
                seen.add(f2[0])
                freqs.append(line1.strip())
                freqs.append(line2.strip())
                line1 = f_freqs.readline()
                line2 = g_freqs.readline()
        freqs = freqs[:joint_vocab_size + 4]
        with open(f'{freqs_path}.txt', 'w') as f_freqs:
            f_freqs.write('\n'.join(freqs))
        wtoi = {
            word.split()[0]: idx for idx, word in enumerate(freqs)
        }

        # Save vocabularly.
        with open(f'{vocab_path}.txt', 'w') as f_vocab:
            f_vocab.write('\n'.join(
                word.split()[0] for idx, word in enumerate(freqs)))

        bpe_toks = {}
        for lang in langs:
            # Apply the BPE codes.
            if not use_cache or not os.path.isfile(f'{bpe_path}.{lang}'):
                logging.info(f'Applying BPE for language {lang}.')
                with open(f'{tok_path}.filtered.{lang}', 'r') as f_in:
                    apply_cmd = [
                        'subword-nmt', 'apply-bpe', '-c', codes_path,
                        '--vocabulary', f'{freqs_path}.txt',
                    ]
                    bpe_sents = subprocess.check_output(
                        apply_cmd, stdin=f_in).decode('utf-8').split('\n')
                bpe_toks[lang] = [
                    ([wtoi['[CLS]']] + [wtoi[word]
                        if word in wtoi else wtoi['[UNK]']
                        for word in sent.split()]
                        + [wtoi['[SEP]']]
                        + [wtoi['[PAD]']] * (
                            self.max_length - len(sent.split()) - 1)
                        )[:self.max_length + 1]
                    for sent in bpe_sents if sent.split()
                ]
                with open(f'{bpe_path}.{lang}', 'wb') as f_bpe:
                    pickle.dump(bpe_toks[lang], f_bpe)
            else:
                logging.info(f'Using previously calculated BPE tokenization '
                             f'for {lang}.')
                with open(f'{bpe_path}.{lang}', 'rb') as f_bpe:
                    bpe_toks[lang] = pickle.load(f_bpe)

        return bpe_toks

    def _save(self, data, valid_size, dtype='int32'):
        """Saves the datasets to HDF5 files."""
        h5path = os.path.join(self.data_dir, self.name)
        for lang in data:
            h5file = f'{h5path}/{lang}.h5'
            logging.info(f'Saving {lang} dataset to {h5file}')
            with h5py.File(h5file, 'w') as f:
                train_ds = f.create_dataset(
                    'train', data=data[lang][:-valid_size], dtype=np.int32)
                train_ds.attrs['dtype'] = dtype
                valid_ds = f.create_dataset(
                    'valid', data=data[lang][-valid_size:], dtype=np.int32)
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
        tok_path = self._tokenize(out_path, langs, use_cache)
        bpe_toks = self._encode(tok_path, langs, joint_vocab_size, use_cache)
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

    def _encode(self, raw_text_path, langs, use_cache=False):
        """
        Encodes sentences listed one per line in file `raw_text_path` as seqs
        of integers indexing into the BERT multilingual vocabulary.
        """
        self._filter_sents(raw_text_path, langs, use_cache)

        # Load saved tokenized data if we cached it during a previous run.
        int_tok_path = os.path.join(self.data_dir, self.name, f'int_tok.pickle')
        if use_cache and os.path.isfile(int_tok_path):
            logging.info(f'Loading BPE tokenized data from {int_tok_path}.')
            try:
                with open(int_tok_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(
                    f'Loading cached BPE tokenized int data failed: {str(e)}.')

        # Load Bert tokenizer.
        logging.info(f'Encoding data as BPE token indices.')

        # WARNING: If you change the tokenizer, then make sure the above
        #          hard-coded bos, eos and pad token indices are correct.
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-cased', do_lower_case=False)

        # Tokenize the sentences in the given files.
        lengths = {}
        ts = {}
        for lang in langs:
            with open(f'{raw_text_path}.filtered.{lang}', 'r') as f:
                logging.info(f'Converting {lang} text to BPE token indices.')
                ts[lang] = [
                    tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(sent))[:self.max_length]
                    for sent in f
                ]
                lengths[lang] = [len(sent) for sent in ts[lang]]

        # Vectors will have length `max_len + 1` to account for BOS.
        max_len = max([ll for lang in langs for ll in lengths[lang]])
        toks = {}
        for lang in langs:
            logging.info(f'Adding BOS, EOS and PAD tokens for {lang}.')
            toks[lang] = [
                ([self.bos] + sent + [self.eos]
                    + [self.pad] * (max_len - len(sent) - 1))[:max_len + 1]
                for sent in ts[lang]
            ]

        # Save vocabulary to file. (It will be called `vocab.txt`.)
        vocab_dir = os.path.join(self.data_dir, self.name)
        tokenizer.save_vocabulary(vocab_dir)

        # Save BPE tokenized data so we do not have to recompute if we rerun.
        with open(int_tok_path, 'wb') as f:
            logging.info(f'Saving BPE tokenized data to {int_tok_path}.')
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


class DropNthTokenCorpus(BertCorpus):
    """
    This is a corpus where every nth word has been dropped. The BOS token
    and the first token of the sentence are never dropped. The remaining
    non-padding tokens are always terminated by a EOS token.

    This keep `n` versions of each sentence where the token dropping
    starts at different offsets.
    """

    def __init__(self,
                 name,
                 n,
                 shuffle=True,
                 max_length=200):
        super().__init__(name, shuffle, max_length)
        self.n = n

    def _subsample(self, toks, lens):
        """
        Discards every nth token from `toks`.
        """
        logging.info(f'Discarding every {self.n}th token.')
        max_len = min([len(toks[lang][0]) for lang in toks])
        new_max_len = (max_len - max_len // self.n) + 1
        new_toks = {lang: [] for lang in toks}
        new_lens = {lang: [] for lang in lens}
        for lang in toks:
            for sent, ll in zip(toks[lang], lens[lang]):
                for k in range(self.n):
                    new_sent = [
                        self.eos if ll + 1 <= i and i <= ll + 2 else w
                        for i, w in enumerate(sent)
                        if ((i - 1) % self.n != k or i == 1)
                    ]
                    new_sent = \
                        new_sent + [self.pad] * (new_max_len - len(new_sent))
                    new_toks[lang].append(new_sent)
                    new_lens[lang].append(
                        ll - (ll + self.n - k - 1) // self.n + int(k == 0))
        return new_toks, new_lens

    def create(self, datafiles, max_size=None, valid_size=0, use_cache=False):
        """
        Create the dataset from `datafiles` by dropping every nth token.
        """
        out_path, langs = self._clean(datafiles, max_size, use_cache)
        toks, lens = self._encode(out_path, langs)
        if self.shuffle:
            self._shuffle_with_lens(toks, lens)
        toks, lens = self._subsample(toks, lens)
        return self._save_with_lens(toks, lens, valid_size, dtype='int32')


class DropRandomPercentCorpus(BertCorpus):
    """
    This is a corpus which contains each sentence from `tok` starting
    with BOS and the first token of the sentence and with `p` percent total
    tokens randomly kept. The rest of the tokens are discarded.

    The indices of discarded tokens agree across languages.
    """

    def __init__(self,
                 name,
                 p,
                 shuffle=True,
                 max_length=200):
        super().__init__(name, shuffle, max_length)
        self.p = p

    def _subsample(self, toks, lens):
        """
        Keep `self.p` percent tokens from every sentence. Removed tokens
        can be padding as well as part of the sentence.
        """
        logging.info(f'Keeping random set of {self.p * 100}% of tokens.')
        max_len = min([len(toks[lang][0]) for lang in toks])
        n = math.ceil(max_len * self.p)
        new_toks = {lang: [] for lang in toks}
        new_lens = {lang: [] for lang in lens}
        lang1, lang2 = tuple(new_toks.keys())
        for sent1, l1, sent2, l2 in zip(toks[lang1], lens[lang1],
                toks[lang2], lens[lang2]):
            indices = list(range(2, max_len))
            random.shuffle(indices)
            indices = indices[:n]
            indices.sort()
            new_sent1 = [sent1[i] for i in indices]
            new_sent2 = [sent2[i] for i in indices]
            for i, c in enumerate(new_sent1):
                if c == self.eos:
                    break
                elif c == self.pad:
                    new_sent1[i] = self.eos
                    new_lens[lang1].append(i - 1)
                    break
            for i, c in enumerate(new_sent2):
                if c == self.eos:
                    break
                elif c == self.pad:
                    new_sent2[i] = self.eos
                    new_lens[lang2].append(i - 1)
                    break
            new_toks[lang1].append(new_sent1)
            new_toks[lang2].append(new_sent2)
        return new_toks, new_lens

    def create(self, datafiles, max_size=None, valid_size=0, use_cache=False):
        """
        Create the dataset from `datafiles` by keeping `p` percent of
        the input/output tokens.
        """
        out_path, langs = self._clean(datafiles, max_size, use_cache)
        toks, lens = self._encode(out_path, langs)
        if self.shuffle:
            self._shuffle_with_lens(toks, lens)
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
            emb = np.array(bert_emb(torch.LongTensor(x)).data, dtype=np.float32)
            if x.shape[0] < chunk_size:
                # This is a technical step to prevent returning too few rows.
                dims = (chunk_size - x.shape[0], max_length, self.emb_size)
                return np.concatenate((emb, np.zeros(dims, dtype=np.float32)))
            return emb

        bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        bert_model.eval()
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
        self.bos_emb = np.array(
            bert_emb(torch.tensor([self.bos])).data[0], dtype=np.float32)
        self.eos_emb = np.array(
            bert_emb(torch.tensor([self.eos])).data[0], dtype=np.float32)
        self.pad_emb = np.array(
            bert_emb(torch.tensor([self.pad])).data[0], dtype=np.float32)
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
            logging.info(f'Saving {lang} dataset to {h5file}')
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
