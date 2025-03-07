ARG PYTHON_VERSION=3.11.8

FROM python:${PYTHON_VERSION}-slim-bookworm as base
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    libgl1 git \
    libglib2.0-0 \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*


WORKDIR /temp
COPY ./src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
    
RUN mkdir -p /app/logs/

COPY ./src .

CMD celery -A celery_tasks.process_folder worker --loglevel=info --concurrency=1