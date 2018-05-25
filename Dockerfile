FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get gcc g++
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git

#FROM python:3.6

#RUN apt-get -y update  && apt-get install -y gcc g++ \
#  gcc-c++ \
#  libpng-dev \
#  apt-utils \
# && rm -rf /var/lib/apt/lists/*

RUN mkdir /final

WORKDIR /final

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt


RUN git clone https://github.com/sjv1030/data602-finalproject ../final/finalproject

EXPOSE 27017
EXPOSE 5000
EXPOSE 8050

CMD [ "python3", "/final/finalproject/index.py" ]
