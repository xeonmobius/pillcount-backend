FROM python:3.7.4

WORKDIR /flask

ADD . /flask

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
