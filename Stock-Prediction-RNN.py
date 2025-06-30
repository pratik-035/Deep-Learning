# Practical 9.1 :  Stock Price Prediction using RNN Implement a simple form of a RNN

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
from torch.autograd import Variable

#Simulate some stock prices 
np.random.seed(42) 
time_steps = np.linspace(0, 100, 400) 
prices = np.sin(time_steps) + np.random.normal(scale=0.5, 
size=len(time_steps))

plt.figure(figsize=(12, 6)) 
plt.plot(time_steps, prices, label='Stock Price') 
plt.title('Simulated Stock Prices') 
plt.xlabel('Time') 
plt.ylabel('Price') 
plt.legend() 
plt.show() 

from sklearn.preprocessing import MinMaxScaler 

#Normalize the data 
scaler = MinMaxScaler(feature_range=(-1, 1)) 
prices_normalized = scaler.fit_transform(prices.reshape(-1, 1)).flatten() 

#Prepare the data for RNN 
def sliding_windows(data, seq_length): 
    x = [] 
    y = [] 
    for i in range(len(data) - seq_length - 1): 
        _x = data[i:(i+seq_length)] 
        _y = data[i+seq_length] 
        x.append(_x)
        y.apend(_y)
    return np.array(_x)

seq_length = 5 
x, y = sliding_windows(prices_normalized, seq_length)

#Split the data into train and test sets 
train_size = int(len(y) * 0.67) 
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x))) 
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size]))) 
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)]))) 
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

#Define a simple RNN model 
class SimpleRNN(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, output_size): 
        super(SimpleRNN, self).__init__() 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) 
        self.fc = nn.Linear(hidden_size, output_size) 
        
        
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        out, _ = self.rnn(x, h0) 
        out = self.fc(out[:, -1, :]) 
        return out
    
num_epochs = 1000
learning_rate = 0.01

input_size = 1 
hidden_size = 2 

num_layers = 1 
output_size = 1

model = SimpleRNN(input_size, hidden_size, num_layers, output_size) 

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

#Train the model 
for epochs in range(num_epochs): 
    outputs = model(trainX.unsqueeze(-1)) 
    optimizer.zero_grad() 
    
    loss = criterion(outputs, trainY) 
    loss.backward() 
    optimizer.step() 
    if epochs % 100 == 0: 
        print("Epoch: %d, loss: %1.5f" % (epochs, loss.item())) 
        
#make predictions 
model.eval() 
train_predict = model(dataX.unsqueeze(-1)) 

#Invert predections 
data_predict = train_predict.data.numpy() 
data_predict = scaler.inverse_transform(data_predict).flatten()

#Invert actual prices 
actual_prices = scaler.inverse_transform(dataY.data.numpy().reshape(-1, 1)).flatten() 

#Plot results 
plt.figure(figsize=(12, 6)) 
plt.axvline(x=train_size, c='r', linestyle='--')

#plotting actual prices 
plt.plot(actual_prices, label='Actual Prices') 
plt.plot(data_predict, label='Predicted Prices') 
plt.title('Stock Price Prediction') 
plt.xlabel('Time') 
plt.ylabel('Price') 
plt.legend() 
plt.show() 
