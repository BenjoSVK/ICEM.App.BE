FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

#

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/logs/

COPY . .


# Initialize and update the submodule
# RUN git submodule update --init --recursive

RUN pip install  -r requirements.txt
RUN pip install  -r src/iedl_segmentation/requirements.txt

CMD python src/app.py