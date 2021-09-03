FROM ubuntu:latest

WORKDIR /flask

ADD . /flask

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3 \
    pip \
    && rm -rf /var/lib/apt/lists/* \
    pip install -r requirements.txt

EXPOSE 5000

CMD ["/usr/local/bin/gunicorn"  , "-b", "0.0.0.0:5000", "main:app", "-w", "4"]