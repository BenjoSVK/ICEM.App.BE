ARG PYTHON_VERSION=3.11.8

FROM python:${PYTHON_VERSION}-slim-bookworm as base
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN mkdir -p /app/logs/

COPY ./src .

RUN pip install --no-cache-dir -r requirements.txt

CMD python app.py