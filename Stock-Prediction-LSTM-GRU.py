# Practical 7: Stock Price Prediction using LSTM and GRU Models.  


# Import Statements 
import yfinance as yf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, GRU, Dense 


# Load stock data from Yahoo Finance 
stock_symbol = 'AAPL' 
df = yf.download(stock_symbol, start="2010-01-01", end="2023-01-01") 


# Use the 'Close' price for modeling 
data = df[['Close']].values 

# Normalise the data 
scaler = MinMaxScaler(feature_range=(0,1)) 
data_scaled = scaler.fit_transform(data) 

#Split the data into train and test sets 
train_size = int(len(data_scaled) * 0.8) 
train, test = data_scaled[0:train_size], data_scaled[train_size:] 

# Convert the data into time-series format 
def create_dataset(dataset, look_back=1): 
    X, Y = [], [] 
    for i in range(len(dataset) - look_back): 
        X.append(data[i:(i+look_back), 0]) 
        Y.append(data[i+look_back, 0]) 
    return np.array(X), np.array(Y) 

look_back = 60 
X_train, y_train = create_dataset(train, look_back) 
X_test, y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features] 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 

# Build LSTM Model 
model_lstm = Sequential() 
model_lstm.add(LSTM(50, input_shape=(look_back, 1))) 
model_lstm.add(Dense(1)) 
model_lstm.compile(loss='mean_squared_error', optimizer='adam') 

#Build GRU Model 
model_gru = Sequential() 
model_gru.add(GRU(50, input_shape=(look_back, 1))) 
model_gru.add(Dense(1)) 
model_gru.compile(loss='mean_squared_error', optimizer='adam') 

# Train the model 
model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1) 
model_gru.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1) 

#Make Predictions 
lstm_predictions = model_lstm.predict(X_test) 
gru_predictions = model_gru.predict(X_test) 

#Inverse transform predictions and actual values 
lstm_predictions = scaler.inverse_transform(lstm_predictions) 
gru_predictions = scaler.inverse_transform(gru_predictions) 
y_test_actual = scaler.inverse_transform([y_test]) 

# Future predictions (next 30 days) 
future_steps = 30 
last_sequence = X_test[-1] #Last sequence in test data 

#Predict future predictions using the LSTM Model 
future_predictions_lstm = [] 
for i in range(future_steps): 
    prediction = model_lstm.predict(np.reshape(last_sequence, (1, 
look_back, 1))) 
    future_predictions_lstm.append(prediction[0][0]) 
    last_sequence = np.append(last_sequence[1:], prediction[0][0]) 
    
    
#Predict future prices using the GRU Model 
last_sequence = X_test[-1] 
future_predictions_gru = [] 
for i in range(future_steps): 
    prediction = model_gru.predict(np.reshape(last_sequence, (1, 
look_back, 1))) 
    future_predictions_gru.append(prediction[0][0]) 
    last_sequence = np.append(last_sequence[1:], prediction[0][0])
    
#Inverse transform future predictions 
future_predictions_lstm = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1)) 
future_predictions_gru = scaler.inverse_transform(np.array(future_predictions_gru).reshape(-1, 1)) 

#Create a new time range for future predictions 
future_range = np.arange(len(y_test_actual[0]), len(y_test_actual[0]) + 
future_steps) 

#Plot the results 
plt.figure(figsize=(14,7)) 
plt.plot(y_test_actual[0], label='Actual Price', color='blue') 
plt.plot(lstm_predictions, label='LSTM Predictions', color='orange') 
plt.plot(gru_predictions, label='GRU Predictions', color='green') 


#Plot the future predictions 
plt.plot(future_range, future_predictions_lstm, label='Future LSTM Predictions', color='red', linestyle='--') 
plt.plot(future_range, future_predictions_gru, label='Future GRU Predictions', color='purple', linestyle='--') 


plt.title(f'{stock_symbol} Stock price prediction and future forecasting - LSTM vs GRU') 
plt.xlabel('Time Steps') 
plt.ylabel('Stock Price') 
plt.legend() 
plt.show() 