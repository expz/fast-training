# Fast Training of Machine Translation Models

This project was developed during an Insight AI Fellowsip in the summer of 2019. It is in two parts. The first part has code for __fast training of a model__ that translates French sentences into English. The second part is a __small webapp__ that translates French sentences to English using a trained model.

The model definition used the Pervasive Attention repo (https://github.com/elbayadm/gttn2d) as a reference implementation. Unless marked with an attribution (see esp. docstrings at the beginning of classes and functions) all code is my own. If you are looking for code to reuse in your own work, here is a list of classes that I would have liked to have found implemented elsewhere instead of writing my own:

* (LanguageCorpus)[https://github.com/expz/fast-training/blob/master/src/corpus.py] and its descendents make a flexible framework for forming HDF5 training sets from raw files of sentences in various ways.

* (SubSampler and DistributedSampler)[https://github.com/expz/fast-training/blob/master/src/dataloader.py] implement samplers that allow epochs to be smaller than the entire training set while (1) still being randomly shuffled once for every pass through the entire dataset (2) not repeating any samples until the current pass through the training set has finished. Most tools are built on the assumption that an epoch does not take long to train, e.g., they only save checkpoints after each epoch, so this adapts a large dataset to that paradigm.

* (`beam_search()`)[https://github.com/expz/fast-training/blob/master/src/evaluate.py] performs a vectorized beam search for the best outputs for a batch of input sentences. It assumes the model works iteratively by initially taking in the input sentence and a beginning of sequence (BOS) token and outputing the first token of the sentence. Then the input sentence and the BOS and first output token are fed into the network to get the second output token, etc.

## Requisites

- Linux or MacOS. Windows Subsystem for Linux (WSL) might work too, but I have not tested it.
- `bash`
- `git`
- `python` (Python version 3.6)
- `pip`
  * For example, on Ubuntu 18.04 run `sudo apt-get install python3-pip`.
- `virtualenv`
  * If everything else is installed, you should be able to install `virtualenv` by running `pip3 install virtualenv`.

If you prefer to use `conda` for package management, then the list of required packages is in `requirements.in` and `dev-requirements.in` (not `requirements.txt` which lists dependencies too).

#### Only for model training

- `perl`

## Setup

1. Clone repository.
```
git clone https://github.com/expz/fast-training
```

2. Create Python environment. __NOTE:__ Moving this directory after creating the Python environment will break the environment and require deleting and redoing the setup.
```
cd fast-training
deactivate 2>/dev/null
virtualenv --python=$(which python3) fast-training
```

3. Install Python packages.
```
source fast-training/bin/activate
source .env
pip install -r requirements.txt
```

## Run webapp

1. Activate Python environment. From the root directory of this repository, run
```
source fast-training/bin/activate
source .env
```

2. Run the app.
```
gunicorn -b 0.0.0.0:8887 app:api
```

3. Navigate to [http://localhost:8887](http://localhost:8887) in a browser to see the website. The health check is available at [http://localhost:8887/health](http://localhost:8887/health). It should print 'OK'.

4. To test the API directly, open a new terminal and run (requires `curl` to be installed):
```
curl -X POST -H 'Content-type: application/json' -d '{"fr": "Bonjour le monde!"}' http://localhost:8887/api/translate/fren
```

### Hosting on Google Cloud Platform

If you would like to host the app on Google Cloud, then set `GCP_PROJECT`, e.g. run
```
export GCP_PROJECT=my-google-cloud-project
```
Then build the docker container and push it to Google Cloud Run:
```
./build/build
```

## Training the model

__WARNING__: Training takes substantial computing resources. Some datasets are large and require significant computing power to preprocess and significant RAM as working space (although preprocessing does not load all data in memory for embedding vector datasets). At least one NVIDIA GPU is also required.

Run the following commands from the root directory of the repository.

1. Load the environment and install development requirements.
```
source .env
pip install -r dev-requirements.txt
```

2. Download and prepare a small dataset of a few hundred MB. For more data, a larger data source or multiple data sources can be used. Possible data sources are listed using `python run.py pepare-data list-datasets`.
```
python run.py prepare-data bert fr bert_fr_en "['news2014']" 50 --shuffle True --valid-size 4096
```

3. (Optional) View the model architecture.
```
python run.py summary config/densenet-12.yaml
```

4. Train a model. With the `bert_fr_en` dataset prepared, a 12 layer Densenet with BERT pretrained embeddings can be trained (`config/densenet-12.yaml`). This can take a long time.
```
python run.py train --lr 0.005 config/densenet-12.yaml
```
Press CTRL-C to stop training in the middle.

## Development

First install development requirements. From the root directory of the repository, run
```
source .env
pip install -r dev-requirements.txt
```

### Test

From the root directory of the repository, run
```
source .env
pytest
```

### Manage requirements

Upgrade packages:
```
pip-compile --upgrade requirements.in
pip-compile --upgrade dev-requirements.in
```

### Format code

This repo uses Google's Yet Another Python Formatter (yapf) to format code.
```
yapf -i --style=google src/file_to_format.py
```
