# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:59:51 2021

@author: va
"""

from cryptocmd import CmcScraper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
import math
import datetime as dt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from keras.models import load_model

def prediction(): 
  scraper = CmcScraper("BTC")

  headers, data = scraper.get_data()

  df = scraper.get_dataframe()


  df.index = sorted(df.index.values, reverse=True)
  df.sort_index(inplace = True)

  df1 = df.reset_index()['Close']

  scaler = MinMaxScaler(feature_range=(0,1))
  df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))


  train_size = int(len(df1)*0.65)
  test_size = len(df1)-train_size
  train_data,test_data = df1[0:train_size,:], df1[train_size:len(df1),:1]

  def dataset(data,time_stemp):
      X_data = []
      Y_data = []
      for i in range(len(data) - time_stemp -1):
          X_data.append(data[i:(i+time_stemp),0])
          Y_data.append(data[i+time_stemp,0])
      return np.array(X_data),np.array(Y_data)

  time_stemp = 100
  X_train,y_train = dataset(train_data,time_stemp)
  X_test,y_test = dataset(test_data,time_stemp)


  X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
  X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)


  model =load_model('Bitcointrial1.h5')


  train_prediction = model.predict(X_train)
  test_prediction = model.predict(X_test)

  train_prediction = scaler.inverse_transform(train_prediction)
  test_prediction = scaler.inverse_transform(test_prediction)

  train_score = math.sqrt(mean_squared_error(y_train,train_prediction))
  test_score = math.sqrt(mean_squared_error(y_test,test_prediction))

  look_back = 100
  trainPredictPlot = np.empty_like(df1)
  trainPredictPlot[:, :] = np.nan
  trainPredictPlot[look_back: len(train_prediction)+ look_back, :] = train_prediction

  testPredictPlot = np.empty_like(df1)
  testPredictPlot[:, :] = np.nan
  testPredictPlot[len(train_prediction) + (look_back*2) +1 :len(df1)-1,:] = test_prediction

 

  x_input = test_data[len(test_data)-100:].reshape(1,-1)
  X_input = list(x_input)
  X_input = X_input[0].tolist()

  output = []
  time_stemp = 100
  day = 0
  days = 30
  while(day<days):
      if(len(X_input)>100):
          x_input = np.array(X_input[1:])
          x_input = x_input.reshape(1,-1)
          x_input = x_input.reshape((1, time_stemp, 1))
          day_pred = model.predict(x_input)
          X_input.extend(day_pred[0].tolist())
          X_input = X_input[1:]
          output.extend(day_pred.tolist())
          day = day +1
      else:
          x_input = x_input.reshape(1,time_stemp,1)
          day_pred = model.predict(x_input)
          X_input.extend(day_pred[0].tolist())
          output.extend(day_pred.tolist())
          day = day+1
      
  ###### TO BE DONE IF PLOTTING WITH ORIGINAL DATASET SO THAT COULD BE MERGED
  new_pred = np.arange(1,101)
  Future_pred = np.arange(len(df1)-1,len(df1)+days)
  Future_output = scaler.inverse_transform(output)


  # If to be merged with test set
  Merge_point_test = np.array(test_prediction[-1,:])
  Future_output_test = np.insert(Future_output, 0, Merge_point_test, axis=0)

  # if to be merged with original dataset
  Merge_point_original = np.array(df['Close'][len(df)-1])
  Future_output_original = np.insert(Future_output, 0, Merge_point_original, axis=0)


  import datetime
  future_dates = []
  for i in range(days):
      date = (dt.datetime.today() + datetime.timedelta(days=i)).strftime("%m-%d-%Y")
      future_dates.append(date)

 
  from requests import Request, Session
  from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
  import json

  url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
  parameters = {
    'start':'1',
    'limit':'5000',
    'convert':'USD'
  }
  headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': 'b0bb28f7-7601-4366-9be1-7932ca0d4617',
  }

  session = Session()
  session.headers.update(headers)

  try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)
    live_price = data['data'][0]['quote']['USD']['price']
  except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)

  output = scaler.inverse_transform(output)
  df1 = scaler.inverse_transform(df1)

  startDate = '04-28-2013'
  from datetime import date
  from datetime import timedelta
  

  today = date.today()

  endDate= today- timedelta(days = 1)
  endDate=endDate.strftime("%m-%d-%Y")
 


  return future_dates,output,live_price,df1,trainPredictPlot,testPredictPlot,startDate,endDate,Future_pred,Future_output_original

    
  
 