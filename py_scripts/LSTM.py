import pandas as pd
import datetime
import numpy as np
import tensorflow as tf
import datetime
import time
import pandas as pd
from math import sqrt
from datetime import timedelta
import random
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import roc_curve

def get_future_data(crypto_data):
    crypto_data = crypto_data.reset_index()
    columns = ['Date', 'nat_gas']
    crypto_data_avg = crypto_data[columns]
    price = crypto_data['nat_gas']
    last = crypto_data_avg.tail(1)['Date'].dt.date
    future = []

    for i in range(35):
        time = last + timedelta(days=1)
        last = time
        future.append(time)

    usage = random.sample(range(int(min(price)), int(max(price)+100)), 35)
    future_array = np.concatenate(future, axis=0)
    d = {'Date': future_array, 'nat_gas': usage}
    df = pd.DataFrame(data=d)
    crypto_data_avg_random = crypto_data_avg.append(df)
    prices = crypto_data_avg_random['nat_gas']

    crypto_data_avg.append(crypto_data_avg_random)

    return future_array, prices

def get_train_data(prices):
    values = prices.values.reshape(-1,1)
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    train_size = int(len(scaled) * 0.7)
    test_size = len(scaled) - train_size
    train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
    return train, test, scaler

def create_dataset(dataset, look_back=35):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def run_LSTM(data):
    tf.reset_default_graph()

    future_array, prices = get_future_data(data)

    train, test, scaler = get_train_data(data['nat_gas'])
    trainX, trainY = create_dataset(train, 35)
    testX, testY = create_dataset(test, 35)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(trainX, trainY, epochs=500, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)

    yhat = model.predict(testX)
    yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
    testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))

    train, test, scaler = get_train_data(prices)
    testX, testfutureY = create_dataset(test, 35)
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    yfuture = model.predict(testX)
    yfuture_inverse = scaler.inverse_transform(yfuture.reshape(-1, 1))

    return yhat_inverse, testY, future_array, yfuture_inverse

def LSTM_prediction(future):

    if future == 'ng':
        data = pd.read_csv("py_data/ng_values.csv")
        data['Date'] =  pd.to_datetime(data['Date'], infer_datetime_format=True)
        data = data.reset_index()
    else:
        data = "data"

    yhat_inverse, testY, future_array, yfuture_inverse = run_LSTM(data)
    yhat = [item for sublist in yhat_inverse for item in sublist]
    test_df = pd.DataFrame({"Date": data['Date'][-35:], "nat_gas": yhat[0:35]})
    residuals = testY - yhat

    yfuture = [item for sublist in yfuture_inverse for item in sublist]
    future_df = pd.DataFrame({"Date": future_array, "nat_gas": yfuture[0:35]})

    all_data = data.append(future_df)

    return test_df, future_df, all_data, residuals
