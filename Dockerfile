FROM python:3.6-slim-stretch

RUN apt-get update \
  && apt-get install -y \
    build-essential \
    curl \
    gcc \
    perl \
  && mkdir -p /app/data/moses/tokenizer \
  && mkdir -p /app/data/moses/share/nonbreaking_prefixes \
  && mkdir -p /app/data/subword_nmt_fr_en \
  && mkdir /app/model \
  && mkdir /app/config

RUN curl -o /app/data/moses/share/nonbreaking_prefixes/nonbreaking_prefix.en \
    https://raw.githubusercontent.com/moses-smt/mosesdecoder/ef028446f3640e007215b4576a4dc52a9c9de6db/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en \
  && curl -o /app/data/moses/share/nonbreaking_prefixes/nonbreaking_prefix.fr \
    https://raw.githubusercontent.com/moses-smt/mosesdecoder/ef028446f3640e007215b4576a4dc52a9c9de6db/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.fr \
  && curl -o /app/data/moses/tokenizer/tokenizer.perl \
    https://raw.githubusercontent.com/moses-smt/mosesdecoder/ef028446f3640e007215b4576a4dc52a9c9de6db/scripts/tokenizer/tokenizer.perl

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY config/fr2en.yaml config/default.yaml /app/config/

COPY src /app/src

COPY entrypoint /app/entrypoint

RUN chmod a+x /app/entrypoint

COPY data/examples-fr.txt /app/data/examples-fr.txt

WORKDIR /app

ENV PORT 8080

ENV MODEL_FILE '/app/model/fr2en.pth'

ENV VOCAB_FILE '/app/data/subword_nmt_fr_en/vocab.txt'

ENV PYTHONPATH '/app/src'

ENTRYPOINT ["/app/entrypoint"]
