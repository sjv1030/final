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
from py_scripts import mongoQueryScripts

"""
@author: michelebradley
"""

def get_future_data(crypto_data, headers):
    crypto_data = crypto_data.reset_index()
    crypto_data_avg = crypto_data[headers]
    price = crypto_data[headers[1]]
    last = crypto_data_avg.tail(1)[headers[0]].dt.date
    future = []

    for i in range(7):
        time = last + timedelta(days=31)
        last = time
        future.append(time)

    usage = random.sample(range(int(min(price)-2), int(max(price)+2)), 7)
    future_array = np.concatenate(future, axis=0)
    d = {headers[0]: future_array, headers[1]: usage}
    df = pd.DataFrame(data=d)
    crypto_data_avg_random = crypto_data_avg.append(df)
    prices = crypto_data_avg_random[headers[1]]

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

def create_dataset(dataset, look_back=7):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def run_LSTM(data, col_name, headers):
    tf.reset_default_graph()

    future_array, prices = get_future_data(data, headers)

    train, test, scaler = get_train_data(data[col_name])
    trainX, trainY = create_dataset(train, 7)
    testX, testY = create_dataset(test, 7)

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
    testX, testfutureY = create_dataset(test, 7)
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    yfuture = model.predict(testX)
    yfuture_inverse = scaler.inverse_transform(yfuture.reshape(-1, 1))

    return yhat_inverse, testY, future_array, yfuture_inverse

def recommendation(last_price, predictions):
    number = len(predictions)
    sell = 0
    buy = 0
    for n in range(number):
        predict = float(predictions.values[n])
        if last_price > predict:
            sell = sell + 1
        else:
            buy = buy + 1
    if sell > buy:
        return "SELL"
    else:
        return "BUY"

def LSTM_prediction(future):

    if future == 'ng':
        ng_df = mongoQueryScripts.ng_df
        data = ng_df.sort_values(by=['month_timestamp'])
        data['month_timestamp'] =  pd.to_datetime(data['month_timestamp'], infer_datetime_format=True)
        data = data.reset_index()
        col_name = "ng_val"
        headers = ["month_timestamp", "ng_val"]
    elif future == "oil":
        wtc_df = mongoQueryScripts.wtc_df
        data = wtc_df.sort_values(by=['month_timestamp'])
        data['month_timestamp'] =  pd.to_datetime(data['month_timestamp'], infer_datetime_format=True)
        data = data.reset_index()
        col_name = "oil_val"
        headers = ["month_timestamp", "oil_val"]
    else:
        print ("choose proper future ng or oil")


    yhat_inverse, testY, future_array, yfuture_inverse = run_LSTM(data, col_name, headers)
    yhat = [item for sublist in yhat_inverse for item in sublist]
    test_df = pd.DataFrame({"month_timestamp": data['month_timestamp'][-7:], col_name: yhat[0:7]})
    residuals = testY - yhat

    yfuture = [item for sublist in yfuture_inverse for item in sublist]
    future_df = pd.DataFrame({"month_timestamp": future_array, col_name: yfuture[0:7]})

    all_data = data.append(future_df)

    recommendations = recommendation(data[col_name].values[-1], future_df[col_name])

    return test_df, future_df, all_data, residuals, recommendations
