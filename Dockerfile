FROM python:3.6

RUN apt-get install python-dev
RUN apt-get -y update  && apt-get install -y \
  libpng-dev \
  apt-utils \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /final

WORKDIR /final

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt


RUN git clone https://github.com/sjv1030/data602-finalproject ../final/finalproject

EXPOSE 27017
EXPOSE 5000
EXPOSE 8050

CMD [ "python3", "/final/finalproject/index.py" ]
