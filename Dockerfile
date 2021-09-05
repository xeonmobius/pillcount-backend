# Get our base image which is a minideb with python
FROM bitnami/python:3.9-prod

# Expose our ports
EXPOSE 5000

# Create our work dir
WORKDIR /pillcount-backend

# Copy files to workdir
COPY . /pillcount-backend

# Run all our shell commands
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \                                
    && pip install -r requirements.txt 

CMD ["gunicorn", "-b", "0.0.0.0:5000", "main:app", "-w", "1"]