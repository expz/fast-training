# Fast Training of Machine Translation Models

This project was developed during an Insight AI Fellowsip in the summer of 2019. It is in two parts. The first part has code for __training of a model__ that translates French sentences into English. The second part is a __small webapp__ that translates French sentences to English using a trained model.

The main point of this project was to test out ideas for speeding up training. Unfortunately none of these ideas worked in practice.

The model definition used the Pervasive Attention repo (https://github.com/elbayadm/gttn2d) as a reference implementation. Unless marked with an attribution (see esp. docstrings at the beginning of classes and functions) all code is my own. If you are looking for code to reuse in your own work, here is a list of classes that I would have liked to have found implemented elsewhere instead of writing my own:

* [LanguageCorpus](https://github.com/expz/fast-training/blob/master/src/corpus.py) and its descendents make a flexible framework for creating training sets from parallel sentences.

* [SubSampler and DistributedSampler](https://github.com/expz/fast-training/blob/master/src/dataloader.py) implement samplers that allow epochs to be smaller than the entire training set.

* [beam_search()](https://github.com/expz/fast-training/blob/master/src/evaluate.py) performs a vectorized beam search for the best outputs for a batch of input sentences.

## Requisites

- Linux or MacOS. Windows Subsystem for Linux (WSL) might work too, but I have not tested it.
- `bash`
- `git`
- `perl` (For third-party tokenization scripts)
- `pip`
  * For example, on Ubuntu 18.04 run `sudo apt-get install python3-pip`.
- `python` (Python version 3.6)
- `virtualenv`
  * If everything else is installed, you should be able to install `virtualenv` by running `pip3 install virtualenv`.

If you prefer to use `conda` for package management, then the list of required packages is in `requirements.in` and `dev-requirements.in` (not `requirements.txt` which lists dependencies too).

## Setup

1. Clone repository.
```
git clone https://github.com/expz/fast-training
```

2. Change directory into the repo.
```
cd fast-training
```

3. Create Python environment named `fast-training`. __NOTE:__ Moving this directory after creating the Python environment will break the environment and require deleting and redoing the setup.
```
deactivate 2>/dev/null
virtualenv --python=$(which python3) fast-training
```

4. Install Python packages.
```
source .env
pip install -r requirements.txt
```

## Translate a sentence

1. Activate Python environment. From the root directory of this repository, run
```
source .env
```

2. Download the tokenizer (19KB), model (32MB) and vocabulary (90KB).
```
./download
```

3. Translate a sentence.
```
python translate.py 'Comment allez-vous?'
```

## Run webapp

1. Activate Python environment. From the root directory of this repository, run
```
source .env
```

2. Download the tokenizer (19KB), model (32MB) and vocabulary (90KB).
```
./download
```

3. Run the app.
```
gunicorn -b 0.0.0.0:8887 app:api
```

4. Navigate to [http://localhost:8887](http://localhost:8887) in a browser to see the website. The health check is available at [http://localhost:8887/health](http://localhost:8887/health). It should print 'OK'.

5. To test the API directly, open a new terminal and run (requires `curl` to be installed):
```
curl -X POST -H 'Content-type: application/json' -d '{"fr": "Comment allez-vous?"}' http://localhost:8887/api/translate/fren
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

__WARNING__: Training takes substantial computing resources. Some datasets are large and require significant computing power to preprocess and significant RAM as working space (although preprocessing does not load all data in memory for embedding vector datasets). The code will run on a CPU, but to run it in a reasonable amount of time at least one NVIDIA GPU is required.

__NOTE__: The possible commands and their syntax can be found by running `python dev.py --help`, or for a subcommand like `prepare-data` it would be `python dev.py prepare-data -- --help`.

Run the following commands from the root directory of the repository.

1. Load the environment and install development requirements.
```
source .env
pip install -r dev-requirements.txt
```

2. Download and prepare a small dataset of a few hundred MB. For more data, a larger data source or multiple data sources can be used. Possible data sources are listed using `python dev.py pepare-data list-datasets`.
```
python dev.py prepare-data standard fr news_fr_en "['news2014']" 50 --shuffle True --joint-vocab-size 24000 --valid-size 4096 --use-cache True
```

3. (Optional) View the model architecture.
```
python dev.py summary config/news_fr_en.yaml
```

4. Train a model. With the `bert_fr_en` dataset prepared, a 6+4 layer Densenet can be trained (`config/news_fr_en.yaml`). This can take a few hours to reach a point of reasonable translations for some sentences. To train on NVIDIA GPU 0 (e.g., if you have just one GPU):
```
python dev.py train config/news_fr_en.yaml --lr 0.01 --device_ids '[0]'
```
To train on multiple GPUs as once (e.g., GPUs 0 and 1), run:
```
python dev.py train config/news_fr_en.yaml --lr 0.01 --device_ids '[0,1]'
```
Press CTRL-C to stop training in the middle. __The model is saved at the end of every epoch in the `model/news_fr_en` folder.__

5. (Optional) To continue training a saved model, e.g., `model/news_fr_en/model_0.pth`, run
```
python dev.py train config/news_fr_en.yaml --lr 0.01 --device_ids '[0]' --restore model/news_fr_en/model_0.pth
```

6. Try out the model. To translate 16 examples, choose a saved model, e.g., `model_0.pth`, and run
```
python dev.py example config/news_fr_en.yaml --gpu_id 0 --model model/news_fr_en/model_0.pth --batch 16
```
Translate a sentence of your own with beam search of beam size 5:
```
python translate.py --config config/news_fr_en.yaml --model model/news_fr_en/model_0.pth --beam 5 'Comment allez-vous?'
```

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
