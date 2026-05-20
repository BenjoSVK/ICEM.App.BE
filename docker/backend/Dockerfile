ARG PYTHON_VERSION=3.11.8

FROM python:${PYTHON_VERSION}-slim-bookworm as base
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libvips-dev \
    pkg-config \
    build-essential \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

WORKDIR /temp
COPY ./src/requirements.txt .

# pyvips sa kompiluje (cffi) – potrebný gcc z build-essential
# torch je veľký (~stovky MB); predvolený pip timeout často spôsobí ReadTimeoutError na pomalej sieti
RUN pip install --default-timeout=1200 --retries=10 --no-cache-dir -r requirements.txt \
    && apt-get update \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y --purge \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/logs/

COPY ./src .

CMD python app.py