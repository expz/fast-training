# Fast Training of Translation Models

## Requisites

- `git`
- `python` (Python version 3.6)
- `pip`
- `virtualenv`

#### Dependencies

- `falcon`
- `gunicorn`

#### Development Dependencies

- `pip-tools`
- `pytest`

## Setup

1. Clone repository.
```
git clone https://github.com/expz/insight
```

2. Create Python environment.
```
cd insight
virtualenv --python=/usr/bin/python3 venv
source venv/bin/activate
source .env
pip install -r requirements.txt
```

3. Run the app.
```
gunicorn app:api
```

4. Navigate to [http://localhost:8000](http://localhost:8000). The health check is available at [http://localhost:8000/health](http://localhost:8000/health). To test the API directly, you can run (requires `curl` to be installed):
```
curl -X POST -H 'Content-type: application/json' -d '{"fr": "Bonjour le monde!"}' http://localhost:8000/api/translate/fren
```

### Hosting on Google Cloud Platform

If you would like to host the app on Google Cloud, then set `GCP_PROJECT`, e.g.,
```
export GCP_PROJECT=my-google-cloud-project
```
Then build the docker container and push it to Google Cloud Run:
```
./build/build
```

## Development

Install development requirements:
```
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
pip-compile --upgrade --generate-hashes requirements.in
pip-compile --upgrade --generate-hashes dev-requirements.in
```
