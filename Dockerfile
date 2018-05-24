# FROM amancevice/pandas:0.22.0-python3-alpine
FROM python:3.6

#RUN apk update && apk upgrade && apk add --no-cache git

#RUN apk add --update curl gcc g++ libpng freetype-dev

RUN mkdir /hw3

WORKDIR /hw3

# RUN pip install numpy pandas pymongo statsmodels flask bokeh json gdax sklearn

# RUN pip install gdax seaborn pymongo matplotlib

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# RUN docker pull mongo

# RUN docker run --name database -p 27017:27017 mongo

RUN git clone https://github.com/sjv1030/data602-finalproject ../final/finalproject

EXPOSE 27017
EXPOSE 5000
EXPOSE 8050

CMD [ "python3", "/final/finalproject/index.py" ]
