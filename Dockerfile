FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/logs/

COPY ./src .

RUN pip install --no-cache-dir -r requirements.txt

CMD python app.py