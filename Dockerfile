FROM python:3.6-alpine3.9

RUN mkdir -p /app

COPY requirements.txt /app

RUN pip install -r /app/requirements.txt

COPY src/ /app

WORKDIR /app

ENV PORT 8080

CMD gunicorn -b 0.0.0.0:$PORT app:api
