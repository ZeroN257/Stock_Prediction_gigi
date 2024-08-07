import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.dates as mdates

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Download stock data
df = yf.download("AAPL", start="2015-01-01", end="2024-04-04")
company = 'Apple'
df.reset_index(inplace=True)

# Load the timestamp data
timestamp_df = pd.read_csv('daily_mentions_apple.csv', parse_dates=['timestamp'], dayfirst=True)

# Rename columns for clarity
timestamp_df.rename(columns={'timestamp': 'Date', 'Apple': 'Mentions'}, inplace=True)

# Merge with stock price data
merged_df = pd.merge(df, timestamp_df, on='Date', how='left')

# Check if the Mentions column is added correctly
print(merged_df.head())

# Fill NaN values in Mentions with 0 (assuming no mentions on those days)
merged_df['Mentions'].fillna(0, inplace=True)

# Select stock price records for the last five years, starting from 2015
data_5years = merged_df[merged_df["Date"].dt.year >= 2015]

# Check filtered data shape
data_5years.shape

# Define selected features and target attribute
features = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Mentions"]
target = "Close"

# Define start and end time for each period
train_end_date = pd.to_datetime("2023-06-30")
validate_start_date = pd.to_datetime("2023-07-01")
validate_end_date = pd.to_datetime("2023-12-31")
test_start_date = pd.to_datetime("2024-01-01")
test_end_date = pd.to_datetime("2024-02-29")

# Split dataset into training, validation, and testing
data_train = data_5years[data_5years["Date"] <= train_end_date][features]
data_train_dates = data_5years[data_5years["Date"] <= train_end_date]["Date"]
data_validate = data_5years[(data_5years["Date"] >= validate_start_date) & (data_5years["Date"] <= validate_end_date)][features]
data_validate_dates = data_5years[(data_5years["Date"] >= validate_start_date) & (data_5years["Date"] <= validate_end_date)]["Date"]
data_test = data_5years[(data_5years["Date"] >= test_start_date) & (data_5years["Date"] <= test_end_date)][features]
data_test_dates = data_5years[(data_5years["Date"] >= test_start_date) & (data_5years["Date"] <= test_end_date)]["Date"]

# Initialize scaler with range [0,1]
sc = MinMaxScaler(feature_range=(0,1))

# Fit and transform scaler to training set
data_train_scaled = sc.fit_transform(data_train)

# Transform validating and testing datasets
data_validate_scaled = sc.transform(data_validate)
data_test_scaled = sc.transform(data_test)

# Combine dates with each corresponding dataset
data_train_scaled_final = pd.DataFrame(data_train_scaled, columns=features, index=None)
data_train_scaled_final["Date"] = data_train_dates.values

data_validate_scaled_final = pd.DataFrame(data_validate_scaled, columns=features, index=None)
data_validate_scaled_final["Date"] = data_validate_dates.values

data_test_scaled_final = pd.DataFrame(data_test_scaled, columns=features, index=None)
data_test_scaled_final["Date"] = data_test_dates.values

# Check loaded datasets shape
data_train_df = data_train_scaled_final
data_validate_df = data_validate_scaled_final
data_test_df = data_test_scaled_final
print(f"Training Dataset Shape: {data_train_df.shape}")
print(f"Validation Dataset Shape: {data_validate_df.shape}")
print(f"Testing Dataset Shape: {data_test_df.shape}")

# Convert Date column to a valid Datetime format
data_train_df["Date"] = pd.to_datetime(data_train_df["Date"])
data_validate_df["Date"] = pd.to_datetime(data_validate_df["Date"])
data_test_df["Date"] = pd.to_datetime(data_test_df["Date"])

# Extract dates from each dataset
data_train_dates = data_train_df["Date"]
data_validate_dates = data_validate_df["Date"]
data_test_dates = data_test_df["Date"]

# Extract features
features = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Mentions"]
data_train_scaled = data_train_df[features].values
data_validate_scaled = data_validate_df[features].values
data_test_scaled = data_test_df[features].values

# Define a method to construct the input data X and Y
def construct_lstm_data(data, sequence_size, target_attr_idx):
    
    # Initialize constructed data variables
    data_X = []
    data_y = []

    # Iterate over the dataset
    for i in range(sequence_size, len(data)):
        data_X.append(data[i-sequence_size:i,0:data.shape[1]])
        data_y.append(data[i,target_attr_idx])

    # Return constructed variables
    return np.array(data_X), np.array(data_y)

# Define the sequence size
sequence_size = 60

# Construct training data
X_train, y_train = construct_lstm_data(data_train_scaled, sequence_size, 0)

# Combine scaled datasets all together
data_all_scaled = np.concatenate([data_train_scaled, data_validate_scaled, data_test_scaled], axis=0)

# Calculate data size
train_size = len(data_train_scaled)
validate_size = len(data_validate_scaled)
test_size = len(data_test_scaled)

# Construct validation dataset
X_validate, y_validate = construct_lstm_data(data_all_scaled[train_size-sequence_size:train_size+validate_size,:], sequence_size, 0)

# Construct testing dataset
X_test, y_test = construct_lstm_data(data_all_scaled[-(test_size+sequence_size):,:], sequence_size, 0)

# Initializing the model
regressor = Sequential()

# Add input layer
regressor.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

# Add first LSTM layer and dropout regularization layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Add second LSTM layer and dropout regularization layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Add third LSTM layer and dropout regularization layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Add forth LSTM layer and dropout regularization layer
regressor.add(LSTM(units = 100))
regressor.add(Dropout(rate = 0.2))

# Add last dense layer/output layer
regressor.add(Dense(units = 1))

# Compiling the model
regressor.compile(optimizer = "adam", loss="mean_squared_error")

# Create a checkpoint to monitor the validation loss and save the model with the best performance.
model_location = "..//models//"
model_name = "Apple_stock_price_lstm.model.keras"
best_model_checkpoint_callback = ModelCheckpoint(
    model_location + model_name,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=0)

# Training the model
history = regressor.fit(
    x = X_train,
    y = y_train,
    validation_data=(X_validate, y_validate),
    epochs=10,
    batch_size = 64,
    callbacks = [best_model_checkpoint_callback])

# Prepare model location and name
model_location = "..//models//"
model_name = "Apple_stock_price_lstm.model.keras"

# Load the best performing model
best_model = load_model(model_location + model_name)

# Predict stock price for all data splits
y_train_predict = best_model.predict(X_train)
y_validate_predict = best_model.predict(X_validate)
y_test_predict = best_model.predict(X_test)

# Restore actual distribution for predicted prices
y_train_inv = sc.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((len(y_train), 6))), axis=1))[:, 0]
y_validate_inv = sc.inverse_transform(np.concatenate((y_validate.reshape(-1, 1), np.zeros((len(y_validate), 6))), axis=1))[:, 0]
y_test_inv = sc.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), 6))), axis=1))[:, 0]

y_train_predict_inv = sc.inverse_transform(np.concatenate((y_train_predict, np.zeros((len(y_train_predict), 6))), axis=1))[:, 0]
y_validate_predict_inv = sc.inverse_transform(np.concatenate((y_validate_predict, np.zeros((len(y_validate_predict), 6))), axis=1))[:, 0]
y_test_predict_inv = sc.inverse_transform(np.concatenate((y_test_predict, np.zeros((len(y_test_predict), 6))), axis=1))[:, 0]

# Define chart colors
train_actual_color = "cornflowerblue"
validate_actual_color = "orange"
test_actual_color = "green"
train_predicted_color = "lightblue"
validate_predicted_color = "peru"
test_predicted_color = "limegreen"

# Plot actual and predicted price
plt.figure(figsize=(18, 6))
plt.plot(data_train_dates[sequence_size:], y_train_inv, label="Training Data", color=train_actual_color)
plt.plot(data_train_dates[sequence_size:], y_train_predict_inv, label="Training Predictions", linewidth=1, color=train_predicted_color)

plt.plot(data_validate_dates, y_validate_inv, label="Validation Data", color=validate_actual_color)
plt.plot(data_validate_dates, y_validate_predict_inv, label="Validation Predictions", linewidth=1, color=validate_predicted_color)

plt.plot(data_test_dates, y_test_inv, label="Testing Data", color=test_actual_color)
plt.plot(data_test_dates, y_test_predict_inv, label="Testing Predictions", linewidth=1, color=test_predicted_color)

plt.title("Apple Stock Price Predictions With LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.legend()
plt.grid(color="lightgray")
plt.show()

import pandas as pd

# Create individual dataframes for training, validation, and test sets
train_df = pd.DataFrame({
    'Date': data_train_dates[sequence_size:], 
    'Actual_Train_Price': y_train_inv,
    'Predicted_Train_Price': y_train_predict_inv
})

validate_df = pd.DataFrame({
    'Date': data_validate_dates,
    'Actual_Validate_Price': y_validate_inv,
    'Predicted_Validate_Price': y_validate_predict_inv
})

test_df = pd.DataFrame({
    'Date': data_test_dates,
    'Actual_Test_Price': y_test_inv,
    'Predicted_Test_Price': y_test_predict_inv
})

# Concatenate the dataframes into a single dataframe
all_data_df = pd.concat([train_df, validate_df, test_df], ignore_index=True)

# Print the first few rows of the dataframe
print(all_data_df.head())
print(all_data_df.tail())

#Access ipynb File for Full code
