FROM ubuntu:16.04

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git
RUN apt-get update

#FROM python:3.6

#RUN apt-get -y update  && apt-get install -y gcc g++ \
#  gcc-c++ \
#  libpng-dev \
#  apt-utils \
# && rm -rf /var/lib/apt/lists/*

RUN mkdir /final

WORKDIR /final

COPY requirements.txt ./

#if using Ubuntu above...try this 
RUN pip install --no-cache-dir -r requirements.txt
#if not using ubuntu above...this this
#RUN pip install --no-cache-dir -r requirements.txt


RUN git clone https://github.com/sjv1030/data602-finalproject ../final/finalproject

EXPOSE 27017
EXPOSE 5000
EXPOSE 8050

CMD [ "python3", "/final/finalproject/index.py" ]
