FROM python:slim

WORKDIR /flask

ADD . /flask

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn"  , "-b", "0.0.0.0:5000", "main:app", "-w", "4"]