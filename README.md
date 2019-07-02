# Fast Training of Machine Translation Models

This project was developed when I was an Insight AI Fellow in the summer of 2019. It is in two parts. The first part has code for __fast training of a model__ that translates French sentences into English. The second part is a __small webapp__ that translates French sentences to English using a trained model.

The model code used the Pervasive Attention repo (https://github.com/elbayadm/attn2d) as a reference implementation, although this implementation uses `fastai` and implements distributed parallel training. Unless marked with an attribution (see esp. docstrings at the beginning of classes and functions) all code is my own.

## Requisites

- Linux or MacOS. Windows Subsystem for Linux (WSL) might work too, but I have not tested it.
- `bash`
- `git`
- `python` (Python version 3.6)
- `pip`
  * For example, on Ubuntu 18.04 run `sudo apt-get install python3-pip`.
- `virtualenv`
  * If everything else is installed, you should be able to install it by running `pip3 install virtualenv`.

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
