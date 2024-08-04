import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_and_process_data(ticker, start_date, end_date, nan_strategy='drop', train_test_split_method='ratio', train_size=0.8, test_size=0.2, split_date=None, random_state=None, save_local=False, local_path='data.csv', scale_features=False, scaler_type='minmax'):
    """
    Load and process stock data with multiple features.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - nan_strategy (str): Strategy to deal with NaN values ('drop' or 'fill').
    - train_test_split_method (str): Method to split data ('ratio', 'date', 'random').
    - train_size (float): Ratio of training data if split by ratio.
    - test_size (float): Ratio of test data if split by ratio.
    - split_date (str): Date to split data if split by date.
    - random_state (int): Random seed for reproducibility.
    - save_local (bool): Whether to save the data locally.
    - local_path (str): Path to save the local data.
    - scale_features (bool): Whether to scale feature columns.
    - scaler_type (str): Type of scaler ('minmax' or 'standard').

    Returns:
    - train_data (DataFrame): Training data.
    - test_data (DataFrame): Testing data.
    - scalers (dict): Dictionary of scalers used (if scaling is applied).
    """

    # Check if local data exists
    if save_local and os.path.exists(local_path):
        data = pd.read_csv(local_path, index_col='Date', parse_dates=True)
    else:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Save locally if needed
        if save_local:
            data.to_csv(local_path)
    
    # Handle NaN values
    if nan_strategy == 'drop':
        data = data.dropna()
    elif nan_strategy == 'fill':
        data = data.fillna(method='ffill').fillna(method='bfill')
    else:
        raise ValueError("nan_strategy must be 'drop' or 'fill'")
    
    # Split the data into train and test sets
    if train_test_split_method == 'ratio':
        train_data, test_data = train_test_split(data, train_size=train_size, test_size=test_size, shuffle=False)
    elif train_test_split_method == 'date' and split_date is not None:
        train_data = data[:split_date]
        test_data = data[split_date:]
    elif train_test_split_method == 'random':
        train_data, test_data = train_test_split(data, train_size=train_size, test_size=test_size, random_state=random_state)
    else:
        raise ValueError("train_test_split_method must be 'ratio', 'date', or 'random'")
    
    scalers = {}
    if scale_features:
        for column in data.columns:
            if scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError("scaler_type must be 'minmax' or 'standard'")
            
            train_data[column] = scaler.fit_transform(train_data[[column]])
            test_data[column] = scaler.transform(test_data[[column]])
            scalers[column] = scaler
    
    return train_data, test_data, scalers

# Example usage:
ticker = 'GOOGL'
start_date = '2010-01-01'
end_date = '2023-01-01'
train_data, test_data, scalers = load_and_process_data(ticker, start_date, end_date, nan_strategy='fill', train_test_split_method='ratio', train_size=0.8, save_local=True, local_path='google_stock_data.csv', scale_features=True, scaler_type='minmax')

print("Train Data:")
print(train_data.head())

print("\nTest Data:")
print(test_data.head())
