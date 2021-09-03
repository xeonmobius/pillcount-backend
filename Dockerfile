FROM python:slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /flask

ADD . /flask

RUN pip install -r requirements.txt

EXPOSE 5000

#CMD ["python", "main.py"]

CMD ["gunicorn"  , "-b", "0.0.0.0:5000", "main:app", "-w", "4"]