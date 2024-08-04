import yfinance as yf
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt

def plot_candlestick(ticker, start_date, end_date, save_chart=False, chart_path='chart/candlestick_chart.png'):
    """
    Fetch stock data and display it as a candlestick chart.

    Parameters:
    - ticker: Stock ticker symbol.
    - start_date: Start date for the data.
    - end_date: End date for the data.
    - save_chart: Whether to save the chart as an image.        
    - chart_path: Path to save the chart image.
    """
    # Fetch the stock data
    df = yf.download(ticker, start=start_date, end=end_date)

    # Ensure the index is a datetime index
    df.index = pd.to_datetime(df.index)

    # Plot the candlestick chart
    mpf.plot(df, type='candle', style='charles', title=f'{ticker} Candlestick Chart', ylabel='Price ($)', volume=True)

    # Save the chart as an image if specified
    if save_chart:
        mpf.plot(df, type='candle', style='charles', title=f'{ticker} Candlestick Chart', ylabel='Price ($)', volume=True, savefig=chart_path)


def plot_boxplots(ticker, start_date, end_date, save_chart=False, chart_path='chart/multiple_boxplots_chart.png'):
    """
    Fetch stock data and display it as multiple boxplot charts.

    Parameters:
    - ticker: Stock ticker symbol.
    - start_date: Start date for the data.
    - end_date: End date for the data.
    - save_chart: Whether to save the chart as an image.
    - chart_path: Path to save the chart image.
    """
    # Fetch the stock data
    df = yf.download(ticker, start=start_date, end=end_date)

    # Ensure the index is a datetime index
    df.index = pd.to_datetime(df.index)

    # Create a boxplot for the 'Open', 'High', 'Low', and 'Close' prices
    plt.figure(figsize=(12, 8))
    df_to_plot = df[['Open', 'High', 'Low', 'Close', 'Adj Close']]
    df_to_plot.boxplot()
    
    plt.title(f'{ticker} Boxplot Chart ({start_date} to {end_date})')
    plt.ylabel('Price ($)')
    plt.xticks([1, 2, 3, 4, 5], ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Adj Close'])
 
    # Save the chart as an image if specified
    if save_chart:
        plt.savefig(chart_path)

    # Show the plot
    plt.show()


def plot_volume_boxplot(ticker, start_date, end_date, save_chart=False, chart_path='chart/volume_boxplot_chart.png'):
    """
    Fetch stock data and display it as a boxplot chart for the trading volume.

    Parameters:
    - ticker: Stock ticker symbol.
    - start_date: Start date for the data.
    - end_date: End date for the data.
    - save_chart: Whether to save the chart as an image.
    - chart_path: Path to save the chart image.
    """
    # Fetch the stock data
    df = yf.download(ticker, start=start_date, end=end_date)

    # Ensure the index is a datetime index
    df.index = pd.to_datetime(df.index)

    # Create a boxplot for the 'Volume'
    plt.figure(figsize=(10, 6))
    df_to_plot = df[['Volume']]
    df_to_plot.boxplot()
    
    plt.title(f'{ticker} Volume Boxplot ({start_date} to {end_date})')
    plt.ylabel('Volume')

    # Save the chart as an image if specified
    if save_chart:
        plt.savefig(chart_path)

    # Show the plot
    plt.show()





# Candle_Stick charrt
plot_candlestick(ticker="AAPL", start_date="2023-01-01", end_date="2023-12-31")
# Boxplot
#plot_boxplots(ticker="AAPL", start_date="2023-01-01", end_date="2023-12-31")   
# Example usage
#plot_volume_boxplot(ticker="AAPL", start_date="2023-01-01", end_date="2023-12-31")