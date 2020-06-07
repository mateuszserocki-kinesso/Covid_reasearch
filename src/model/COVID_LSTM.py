import pandas as pd
import numpy as np
from pathlib import Path, PurePath
# from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf  # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
#
# rolling_diff(data_clear, ['Confirmed', 'Deaths', 'Recovered'], ['Continent', 'Country/Region','Province/State'])
def rolling_diff(data_frame, vars_to_calc, vars_to_groupby= ['Continent', 'Country/Region','Province/State']):
    list_of_frames=[]
    core_df = data_frame.copy()
    if len(vars_to_groupby) == 1:
        vars_to_groupby = vars_to_groupby[0]
    for continent in core_df[vars_to_groupby[0]].drop_duplicates():
        # print(continent)
        cont_df = core_df[core_df[vars_to_groupby[0]]==str(continent)].copy()
        for country in cont_df[vars_to_groupby[1]].drop_duplicates():
            country_df = cont_df[cont_df[vars_to_groupby[1]]==country]
            for province in country_df[vars_to_groupby[2]].drop_duplicates():
                prov_df = country_df[country_df[vars_to_groupby[2]] == str(province)].sort_values('Date')
                for variable in vars_to_calc:
                    #print(variable)
                    prov_df["new_cases_" + variable] = 0
                    diff_val = prov_df[variable].copy().values
                    diff_val[1:] -= diff_val[:-1]
                    prov_df["new_cases_" + variable] = diff_val
                    prov_df["new_cases_" + variable][prov_df["new_cases_" + variable] <= 0] = 0
                list_of_frames.append(prov_df.set_index('Date'))
    results = pd.concat(list_of_frames)
    return results
def rolling_diff(data_frame, vars_to_calc, vars_to_groupby):
    core_df = data_frame.copy()
    if len(vars_to_groupby) == 1:
        vars_to_groupby = vars_to_groupby[0]
    for variable in vars_to_calc:
        core_df["new_cases_" + variable] = 0
        for country in core_df[vars_to_groupby[0]].drop_duplicates():
            for province in core_df[vars_to_groupby[1]][core_df[vars_to_groupby[0]]==str(country)].drop_duplicates():
                diff_val = core_df[variable][
                    (core_df[vars_to_groupby[0]] == country) & (core_df[vars_to_groupby[1]] == province)].values
                diff_val[1:] -= diff_val[:-1]
                core_df["new_cases_" + variable][
                    (core_df[vars_to_groupby[0]] == country) & (core_df[vars_to_groupby[1]] == province)] = diff_val
        core_df["new_cases_" + variable][core_df["new_cases_" + variable] <= 0] = 0
    return core_df

p = PurePath(Path.cwd())
path = p.parents[1].joinpath('data')

wd_data = pd.read_csv(path.joinpath('worldometer_data.csv'))
covid_data = pd.read_csv(path.joinpath('covid_19_clean_complete.csv'))
pd.set_option('max_columns', 23)

data = wd_data.merge(covid_data, how='inner', on='Country/Region')

data.columns
data_sorted = data[
    ['Date', 'Continent', 'Country/Region', 'Population', 'Province/State', 'Lat', 'Long', 'WHO Region', 'TotalCases',
     'Serious,Critical', 'Tot Cases/1M pop', 'Deaths/1M pop',
     'TotalTests', 'Tests/1M pop',
     'ActiveCases', 'Confirmed', 'Deaths', 'Recovered']].sort_values(
    ['Continent', 'Country/Region', 'Date']).reset_index(drop=True)
data_sorted['Date'] = pd.to_datetime(data_sorted['Date'])
data_sorted["Week"] = data_sorted['Date'].dt.week
data_sorted["Month"] = data_sorted['Date'].dt.month
data_sorted["Day"] = data_sorted['Date'].dt.day

data_sorted.isnull().sum()

data_clear = data_sorted[(~pd.isnull(data_sorted['Continent'])) & (~pd.isnull(data_sorted['Population']))]

data_clear.isnull().sum()

data_clear['Province/State'] = data_clear['Province/State'].replace(np.nan, 'no_province/state')
data_clear.isnull().sum()
data_clear[data_clear['Country/Region'] == str('Poland')].head(50)

data_new_cases = rolling_diff(data_clear, ['Confirmed', 'Deaths', 'Recovered'], ['Country/Region','Province/State'])

data_new_cases.to_csv(path.joinpath('data_new_cases.csv'))

data_new_cases = pd.read_csv(path.joinpath('data_new_cases.csv'))
data_new_cases.isnull().sum()

data_new_cases.loc[pd.isnull(data_new_cases['ActiveCases']) == True, 'ActiveCases'] = data_new_cases['Confirmed'] - \
                                                                                      data_new_cases['Recovered'] - \
                                                                                      data_new_cases['Deaths']
data_new_cases.isnull().sum()

data_new_cases['deaths_per_m'] = data_new_cases['Deaths'] / 1000000 * data_new_cases['Population'] / 1000000
data_new_cases['TotalTests'] = data_new_cases['TotalTests'].replace(np.nan, -1)
data_new_cases['Tests/1M pop'] = data_new_cases['Tests/1M pop'].replace(np.nan, -1)
data_new_cases['Deaths/1M pop'] = data_new_cases['Deaths/1M pop'].replace(np.nan, -1)
data_new_cases['Serious,Critical'] = data_new_cases['Serious,Critical'].replace(np.nan, -1)
total_sum_df = data_new_cases[['Continent', 'Country/Region', 'Recovered', 'Confirmed', 'Deaths']].groupby(
    ['Continent', 'Country/Region']).sum().reset_index().rename(columns={"Recovered": "Total_recovered",
                                                                         "Confirmed": "Total_confirmed",
                                                                         "Deaths": "Total_deaths"})
#
total_df = data_new_cases.merge(total_sum_df, how='inner', on=['Continent', 'Country/Region'])#.drop('Unnamed: 0',axis=1)
total_df.isnull().sum()
#
total_df['Province/State'] = total_df['Province/State'].replace(np.nan, 'no_province/state')
total_df.isnull().sum()
#
pd.set_option('max_rows',10)
#
one_continent = total_df[total_df['Continent'] == str('Europe')].sort_values('Confirmed').reset_index(
    drop=True).sort_values('Date').drop(['Unnamed: 0'],axis=1).reset_index(drop=True)
one_continent.columns
unique_dates = one_continent['Date'].drop_duplicates().reset_index(drop=True)
split = int(len(unique_dates) * 0.85)
split_date = unique_dates[split]
#
to_lstm = pd.get_dummies(one_continent)
#
to_lstm.columns

#
sorted_var = ['Deaths']
dependent_var = ['Deaths']
all_variables = [column for column in to_lstm.columns if column not in dependent_var ]
sorted_var.extend(all_variables)
to_lstm_sorted = to_lstm[sorted_var]
to_lstm_sorted['Date'].drop_duplicates().reset_index(drop=True)
to_lstm_sorted.columns
# normalize the dataset
scaler_dep = StandardScaler()
scaler_dep.fit(to_lstm_sorted[dependent_var])
scaler = StandardScaler()
scaler.fit(to_lstm_sorted)
dataset = scaler.transform(to_lstm_sorted.reset_index(drop=True).values)  # .reshape(-1, 2)
#
dataset.shape
train_data = dataset[:split]
test_data = dataset[split:]

# train_data = train_data.values.reshape(-1, 1)
# test_data = test_data.values.reshape(-1, 1)

look_back = 7


# tmp = train_data[-4:]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# create and fit the LSTM network
# tx,ty = create_dataset(tmp,look_back)
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)
len(test_data)
train_data.shape
#
trainX.shape
testX.shape
#
trainX_rsp = numpy.reshape(trainX, (trainX.shape[0], train_data.shape[1], trainX.shape[1]))
testX_rsp = numpy.reshape(testX, (testX.shape[0], test_data.shape[1], testX.shape[1]))
#
model = Sequential()
model.add(layers.LSTM(128, input_shape=(trainX_rsp.shape[1], look_back), return_sequences=True))
model.add(layers.LSTM(64,return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX_rsp, trainY, epochs=100, batch_size=512, verbose=1)

# trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=7, verbose=2)

# make predictions
trainPredict = model.predict(trainX_rsp)
testPredict = model.predict(testX_rsp)
# invert predictions
trainPredict = scaler_dep.inverse_transform(trainPredict)
trainY_inv = scaler_dep.inverse_transform([trainY])
testPredict = scaler_dep.inverse_transform(testPredict)
testY_inv = scaler_dep.inverse_transform([testY])
# calculate root mean squared error
#
err_rmse_train = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (err_rmse_train))
err_rmse_test = math.sqrt(mean_squared_error(testY_inv[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (err_rmse_test))
#
err_mae_train = mean_absolute_error(trainY_inv[0], trainPredict[:, 0])
print('Train Score: %.2f MAE' % (err_mae_train))
err_mae_test = mean_absolute_error(testY_inv[0], testPredict[:, 0])
print('Test Score: %.2f MAE' % (err_mae_test))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan

trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
len(trainPredictPlot)
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
len(testPredictPlot)
testPredictPlot[len(trainPredict) + look_back - 1:len(trainPredict) + look_back - 1 + len(testPredict), :] = testPredict
len(testPredict)
len(testPredict)
# plot baseline and predictions
plt.plot()
results = to_lstm_sorted[dependent_var]
res = [x[0] for x in results]

plt.plot(results)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



#OLD ONE


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(one_country[['Deaths', 'Confirmed']].reset_index(drop=True).values.reshape(-1, 1))
one_country['new_cases_Deaths']
one_country.shape[1]
dataset = scaler.transform(one_country.reset_index(drop=True).values)  # .reshape(-1, 2)
#
dataset.shape
train_data = dataset[:split]
test_data = dataset[split:]

# train_data = train_data.values.reshape(-1, 1)
# test_data = test_data.values.reshape(-1, 1)

look_back = 1


# tmp = train_data[-4:]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# create and fit the LSTM network
# tx,ty = create_dataset(tmp,look_back)
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)
#
trainX.shape
testX.shape
#
trainX = numpy.reshape(trainX, (trainX.shape[0], 2, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 2, testX.shape[1]))
#
model = Sequential()
model.add(LSTM(4, input_shape=(2, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=14, verbose=2)

# trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=7, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
#
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
#
trainScore = mean_absolute_error(trainY[0], trainPredict[:, 0])

print('Train Score: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(testY[0], testPredict[:, 0])
print('Test Score: %.2f MAE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
len(trainPredictPlot)
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
len(testPredictPlot)
testPredictPlot[len(trainPredict) + look_back - 1:len(trainPredict) + look_back - 1 + len(testPredict), :] = testPredict
len(testPredict)
len(testPredict)
# plot baseline and predictions
plt.plot()
results = scaler.inverse_transform(dataset)
res = [x[0] for x in results]

plt.plot(res)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
