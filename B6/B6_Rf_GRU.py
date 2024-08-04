import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Download data
df = yf.download("AAPL", start="2015-01-01", end="2024-01-01")
company = 'Apple'
stock_data = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
# Create a new dataframe with relevant features
data = stock_data
data.index = pd.to_datetime(data.index)
train_data, test_data = data[0:int(len(data)*0.9)], data[int(len(data)*0.9):]
# Prepare the data
feature_columns = ['Open', 'High', 'Low', 'Volume']  # Example features
target_column = 'Close'
# Create feature and target arrays
features = data[feature_columns]
target = data[target_column]
# Split data into train and test sets
train_size = int(len(data) * 0.9)
train_features = features.iloc[:train_size]
test_features = features.iloc[train_size:]
train_target = target.iloc[:train_size]
test_target = target.iloc[train_size:]

# Define parameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
# Initialize RandomForestRegressor
rf_model = RandomForestRegressor(random_state=0)
# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# Fit GridSearchCV
grid_search.fit(train_features, train_target)
# Get the best model
best_rf_model = grid_search.best_estimator_
# Make predictions
rf_predictions = best_rf_model.predict(test_features)
# Evaluate the model
mse = mean_squared_error(test_target, rf_predictions)
mae = mean_absolute_error(test_target, rf_predictions)
rmse = math.sqrt(mse)
print('Best parameters found: ', grid_search.best_params_)
print('MSE: ' + str(mse))
print('MAE: ' + str(mae))
print('RMSE: ' + str(rmse))

# Plot results
plt.figure(figsize=(16,8))
plt.plot(data.index[-len(test_features):], test_target, color='red', label='Real Stock Price')
plt.plot(data.index[-len(test_features):], rf_predictions, color='blue', label='Predicted Stock Price')
plt.title('Apple Stock Price Prediction by Random Forest')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.savefig('random_forest_prediction.pdf')
plt.show()


# GRU model
train = train_data.iloc[:, 0:1].values
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
timesteps = 7
X_train = []
y_train = []
for i in range(timesteps, train.shape[0]):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(GRU(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.20))
model.add(GRU(units = 50, return_sequences = True))
model.add(Dropout(0.25))
model.add(GRU(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(GRU(units = 50))
model.add(Dropout(0.25))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 10, batch_size = 32)
real_stock_price = test_data.iloc[:,0:1].values
combine = pd.concat((train_data['Close'], test_data['Close']), axis = 0)
test_inputs = combine[len(combine) - len(test_data) - timesteps:].values
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)

X_test = []
for i in range(timesteps, test_data.shape[0]+timesteps):
    X_test.append(test_inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.figure(figsize=(16,8))
plt.plot(data.index[-600:], data['Close'].tail(600), color='green', label='Train Stock Price')
plt.plot(test_data.index, real_stock_price, color='red', label='Real Stock Price')
plt.plot(test_data.index, predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Apple Stock Price Prediction by GRU')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.savefig('gru_30.pdf')
plt.show()

mse = mean_squared_error(real_stock_price, predicted_stock_price)
print('MSE: '+str(mse))
mae = mean_absolute_error(real_stock_price, predicted_stock_price)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print('RMSE: '+str(rmse))

# Ensure predictions are aligned with test_data index
ran_pre = rf_predictions
gru_predictions = pd.Series(predicted_stock_price.flatten(), index=test_data.index)

# Compute ensemble predictions (simple average)
ensemble_predictions = (ran_pre + gru_predictions) / 2

# Create DataFrame
results_df = pd.DataFrame({
    'Date': test_data.index,
    'Real Stock Price': real_stock_price.flatten(),
    'Random Forest': ran_pre,
    'GRU Prediction': gru_predictions,
    'Ensemble Prediction': ensemble_predictions
})
print(results_df)

# Set Date as index
results_df.set_index('Date', inplace=True)

plt.figure(figsize=(16,8))
plt.plot(data.index[-600:], data['Close'].tail(600), color='black', label='Train Stock Price')
plt.plot(results_df.index, results_df['Real Stock Price'], color='red', label='Real Stock Price')
plt.plot(results_df.index, results_df['Random Forest'], color='green', linestyle='--', label='Random Forest')
plt.plot(results_df.index, results_df['GRU Prediction'], color='blue', linestyle='--', label='GRU Prediction')
plt.plot(results_df.index, results_df['Ensemble Prediction'], color='purple', linestyle='--', label='Ensemble Prediction')

# Formatting x-axis for dates
import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

plt.title('Stock Price Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ensemble_model_comparison.pdf')
plt.show()
