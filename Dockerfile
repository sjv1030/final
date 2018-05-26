FROM python:3.6-slim

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y

# Libs required for geospatial libraries on Debian...
RUN apt-get -y install apt-utils binutils libproj-dev gdal-bin build-essential
# Operational packages .....
RUN apt-get -y install curl wget nano g++ vim libapache2-mod-wsgi

RUN apt-get install --yes gcc libatlas-base-dev gfortran libeigen3-dev

RUN apt-get install -y python3-dev

#RUN apt-get -y update  && apt-get install -y gcc g++ \
#  libpng-dev \
#  apt-utils \
# && rm -rf /var/lib/apt/lists/*

RUN mkdir /final

WORKDIR /final

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/sjv1030/final ../final/finalproject

EXPOSE 27017
EXPOSE 5000
EXPOSE 8050

CMD [ "python3", "/final/finalproject/index.py" ]
