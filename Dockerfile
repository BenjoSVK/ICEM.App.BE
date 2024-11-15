FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN mkdir -p /app/logs/

COPY . .

RUN pip install  -r requirements.txt

CMD python src/app.py