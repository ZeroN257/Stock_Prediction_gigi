#Complete Ver
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN
import matplotlib.pyplot as plt
import yfinance as yf
plt.style.use('fivethirtyeight')  # Set the style of the plots to 'fivethirtyeight'

def create_model(layer_name, num_layers, units_per_layer):
    '''
    Create a deep learning model based on specified parameters.
    
    Parameters:
        layer_name (str): Type of the layers to add ('LSTM', 'GRU', 'RNN')
        num_layers (int): Number of layers to add to the model
        units_per_layer (list): Number of units in each layer
        input_shape (tuple): Shape of the input data (e.g., (x_train.shape[1], 1))
    
    Returns:
        model (keras.Sequential): Compiled deep learning model
    '''
    model = Sequential()  # Initialize the model as a sequential model
    print(f"Creating a model with {num_layers} {layer_name} layers and units per layer as {units_per_layer}")
    for i in range(num_layers):  # Loop through the number of layers
        return_sequences = True if i < num_layers - 1 else False  # Set return_sequences based on layer position
        # Add the specified layer type with the given number of units, managing input shapes and sequence returns
        if layer_name == 'LSTM':
            model.add(LSTM(units_per_layer[i], return_sequences=return_sequences))
        elif layer_name == 'GRU':
            model.add(GRU(units_per_layer[i], return_sequences=return_sequences))
        elif layer_name == 'RNN':
            model.add(SimpleRNN(units_per_layer[i], return_sequences=return_sequences))
    
    model.add(Dense(1))  # Add a dense layer for output with 1 unit
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model with Adam optimizer and MSE loss
    print("Model compiled with optimizer 'adam' and loss 'mean_squared_error'")
    return model

# Download stock data from Yahoo Finance
print("Downloading AAPL stock data...")
df = yf.download("AAPL", start="2012-01-01", end="2024-01-01")  # Download Apple's stock data from 2012 to 2024
print("Data download complete.")

# Process the data
data = df.filter(['Close'])  # Filter out only the 'Close' prices
print("Filtered Close prices from data.")
dataset = data.values  # Convert the filtered data into a numpy array
scaler = MinMaxScaler(feature_range=(0,1))  # Initialize a MinMaxScaler to scale data between 0 and 1
scaled_data = scaler.fit_transform(dataset)  # Scale the data
print("Data scaled to range 0 to 1.")

# Prepare the training data
train_data = scaled_data[0:math.ceil(len(dataset) * .8), :]  # Determine the training data length as 80% of the total
x_train = []  # Initialize x_train for storing features
y_train = []  # Initialize y_train for storing targets
for i in range(60, len(train_data)):  # Create sequences of 60 days as features and the next day as target
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)  # Convert lists to numpy arrays
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape x_train for the LSTM input
print(f"Training data prepared with {len(x_train)} samples.")

# Create, compile, and train the model
model = create_model('LSTM', 2, [50, 50])  # Create a model using the function defined above
print("Training model...")
model.fit(x_train, y_train, batch_size=1, epochs=1)  # Train the model on the training data
print("Model training complete.")


# Split the dataset into training and testing sets
train_data_len = math.ceil(len(dataset) * .8)
test_data = scaled_data[train_data_len - 60:, :]  # Include 60 previous days for test sequences

# Create the test data sequences
x_test = []
y_test = dataset[train_data_len:, :]  # Keep the actual closing prices for the test set
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)  # Convert to numpy array
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape for the LSTM input

# Make predictions on the test data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Inverse transform to get actual values

# Evaluate the model
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))  # Calculate Root Mean Squared Error (RMSE)
print(f"Root Mean Squared Error: {rmse}")

# Plot the data
train = data[:train_data_len]
valid = data[train_data_len:]
valid['Predictions'] = predictions

# Visualize the predictions
plt.figure(figsize=(16,8))
plt.title('Model Evaluation')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.show()

# Display the valid data with predictions
print(valid.head())
