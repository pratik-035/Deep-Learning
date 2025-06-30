# Using PyTorch (Stock Prediction)

import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
from sklearn.preprocessing import MinMaxScaler 

# Detect GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device: {device}") 

# 1. Simulate Stock Prices 
np.random.seed(42) 
time_steps = np.linspace(0, 100, 400) 
prices = np.sin(time_steps) + np.random.normal(scale=0.5, 
size=len(time_steps))

# 2. Visualize Data 
plt.figure(figsize=(10, 6)) 
plt.plot(time_steps, prices, label='Stock Price') 
plt.title('Simulated Stock Prices') 
plt.xlabel('Time') 
plt.ylabel('Price') 
plt.legend() 
plt.show()

# 3. Normalize Data 
scaler = MinMaxScaler(feature_range=(-1, 1)) 
prices_normalized = scaler.fit_transform(prices.reshape(-1, 1))

# 4. Prepare Data (Sliding Window) 
def sliding_windows(data, seq_length): 
    x, y = [], [] 
    for i in range(len(data) - seq_length): 
        x.append(data[i:i+seq_length]) 
        y.append(data[i+seq_length]) 
    return np.array(x), np.array(y) 

seq_length = 10  # Increased sequence length for better learning 
x, y = sliding_windows(prices_normalized, seq_length)

# Convert to PyTorch tensors and move to GPU 
train_size = int(len(y) * 0.67) 
test_size = len(y) - train_size

batch_size = 64  # Increase batch size for better GPU utilization 
# Create training and test sets 
trainX = torch.Tensor(x[:train_size]).view(-1, seq_length, 1).to(device) 
trainY = torch.Tensor(y[:train_size]).to(device)

testX = torch.Tensor(x[train_size:]).view(-1, seq_length, 1).to(device) 
testY = torch.Tensor(y[train_size:]).to(device) 

fullX = torch.Tensor(x).view(-1, seq_length, 1).to(device)  # For full dataset prediction

# 5. Define Optimized LSTM Model 
class OptimizedLSTM(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, output_size): 
        super(OptimizedLSTM, self).__init__() 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        
        # Enable cuDNN optimizations 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=False).to(device) 
 
        self.fc = nn.Linear(hidden_size, output_size) 
        
    def forward(self, x): 
            h0 = torch.zeros(self.num_layers, x.size(0), 
                             self.hidden_size).to(device).detach() 
            c0 = torch.zeros(self.num_layers, x.size(0), 
                             self.hidden_size).to(device).detach() 
            
            # Enable cuDNN optimization 
            with torch.backends.cudnn.flags(enabled=True): 
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])  # Use last time step output return out

# 6. Set Hyperparameters 
input_size = 1 
hidden_size = 128  # Increased hidden size 
num_layers = 3  # More layers for better learning 
output_size = 1 
num_epochs = 500  # Lower epochs due to larger batch size 
learning_rate = 0.001

# 7. Initialize Model, Loss, and Optimizer 
model = OptimizedLSTM(input_size, hidden_size, num_layers, 
output_size).to(device) 
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Use mixed precision training (automatic mixed precision AMP) 
scaler_amp = torch.cuda.amp.GradScaler()

# 8. Train Model with Mixed Precision 
for epoch in range(num_epochs): 
    model.train() 
    optimizer.zero_grad()
    
    # Use automatic mixed precision for faster computation 
    with torch.cuda.amp.autocast(): 
        outputs = model(trainX) 
        loss = criterion(outputs, trainY) 
        
    # Scale loss and backward pass 
    scaler_amp.scale(loss).backward() 
    
    # Clip gradients to avoid explosion 
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
    
    # Update weights 
    scaler_amp.step(optimizer) 
    scaler_amp.update() 
    
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}")
        
# 9. Evaluate Model 
model.eval() 
with torch.no_grad(): 
    train_predict = model(trainX) 
    test_predict = model(testX) 
    full_predict = model(fullX)  # Predict for the entire dataset
    
# 10. Move Predictions Back to CPU 
train_predict = scaler.inverse_transform(train_predict.cpu().numpy()) 
test_predict = scaler.inverse_transform(test_predict.cpu().numpy()) 
full_predict = scaler.inverse_transform(full_predict.cpu().numpy())

actual_prices = scaler.inverse_transform(y)

# 11. Plot Results 
plt.figure(figsize=(10, 6)) 
plt.axvline(x=train_size, c='r', linestyle='--', label='Train/Test Split') 

# Plot actual prices 
plt.plot(actual_prices, label='Actual Price', color='blue') 

# Plot full dataset predictions 
plt.plot(full_predict, label='Predicted Price (Full Dataset)', color='orange')

plt.title('Optimized LSTM Stock Price Prediction (GPU Accelerated)') 
plt.xlabel('Time') 
plt.ylabel('Stock Price') 
plt.legend() 
plt.show() 

    