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
git clone https://github.com/expz/mt-fast-training
```

2. Create Python environment. __NOTE:__ Moving this directory after creating the Python environment will break the environment and require deleting and redoing the setup.
```
cd mt-fast-training
deactivate 2>/dev/null
virtualenv --python=$(which python3) mt-fast-training
```

3. Install Python packages.
```
source mt-fast-training/bin/activate
source .env
pip install -r requirements.txt
```

## Run webapp

1. Activate Python environment. From the root directory of this repository, run
```
source mt-fast-training/bin/activate
source .env
```

2. Run the app.
```
gunicorn -b 0.0.0.0:8887 app:api
```

3. Navigate to [http://localhost:8887](http://localhost:8887). The health check is available at [http://localhost:8887/health](http://localhost:8887/health). To test the API directly, you can run (requires `curl` to be installed):
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

If you have a GPU installed on your system, then you can do training.

Run the following commands from the root directory of the repository.

1. Download and prepare the dataset. This requires over a GB of disk space.
```
./build/prepare-wmt14 --small
```
To prepare the full dataset, omit the `--small`. Preparation for the full dataset can take 15GB of disk space and 10-15 minutes on a fast computer. 

2. Activate the virtual environment if it is not already activated.
```
source mt-fast-training/bin/activate
source .env
```

3. (Optional) View the model architecture.
```
python fr2en.py summary config/densenet-l12.yaml
```

4. (Optional) Search for a good learning rate.
```
streamlit run fr2en.py find_lr config/densenet-l12.yaml
```
Then go in the browser to http://localhost:8501 where a plot of the learning rate performance will appear. Choose the learning rate where the graph has the steepest decline. Then back in the console, press CTRL-C to exit.

5. Train a model. For example, choose the 12 layer Densenet (`config/densenet-l12.yaml`). This can take a long time.
```
python fr2en.py train --lr 0.0015 config/densenet-l12.yaml
```
To do distributed training on nVidia GPUs `0` and `1`, run
```
python fr2en.py train --gpu_ids '[0,1]' config/densenet-l12.yaml
```
Press CTRL-C to stop training in the middle.

## Development

First install development requirements. From the root directory of the repository, run
```
source mt-fast-training/bin/activate
pip install -r dev-requirements.txt
```

### Test

From the root directory of the repository, run
```
source mt-fast-training/bin/activate
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
