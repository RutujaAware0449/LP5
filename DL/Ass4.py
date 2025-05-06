#step 1:import important libraries
# Suppress warnings and logs
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# TensorFlow/Keras modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

#step 2:Load and visualize the dataset
# Load Google stock dataset (GOOGL.csv)
df = pd.read_csv("GOOGL.csv")
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot closing prices
df['Close'].plot(title="Google Stock Closing Price", figsize=(10, 4))
plt.grid(True)
plt.show()

#step 3:prepare the data
# Normalize closing prices
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

# Create sequences
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#Step 4: Build the RNN Model
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

#Step 5: Train the Model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

#Step 6: Make Predictions and Forecast
# Evaluate and predict
loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss:.6f}")

predicted = model.predict(X_test)
actual = scaler.inverse_transform(y_test)
predicted_prices = scaler.inverse_transform(predicted)

# Plot predictions
plt.figure(figsize=(10, 4))
plt.plot(actual, label='Actual')
plt.plot(predicted_prices, label='Predicted')
plt.title("Stock Price Prediction")
plt.legend()
plt.grid(True)
plt.show()

# Predict next day
last_seq = scaled_data[-60:].reshape(1, 60, 1)
next_day_price = scaler.inverse_transform(model.predict(last_seq))
print(f"Predicted Next Day Price: ${next_day_price[0][0]:.2f}")
