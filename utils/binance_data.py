import json
import time
import requests
import pandas as pd
from datetime import datetime

def get_historical_klines(symbol, interval, start_str, end_str, last_timestamp=None):
    """Get historical klines/candlestick data from Binance API.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Kline interval (e.g., '1d', '1h', '15m')
        start_str (str): Start date in 'YYYY-MM-DD' format
        end_str (str): End date in 'YYYY-MM-DD' format
        last_timestamp (int, optional): Last timestamp to continue from previous data
    
    Returns:
        pandas.DataFrame: DataFrame with historical price data
    """
    root_url = 'https://api.binance.com/api/v3/klines'
    
    # Convert strings to timestamps in milliseconds
    start_ts = last_timestamp if last_timestamp else int(datetime.strptime(start_str, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, '%Y-%m-%d').timestamp() * 1000)
    
    all_data = []
    while start_ts < end_ts:
        url = (f"{root_url}?symbol={symbol}&interval={interval}"
               f"&startTime={start_ts}&endTime={end_ts}&limit=1000")
        
        data = json.loads(requests.get(url).text)
        if not data:
            break
            
        all_data.extend(data)
        
        # Update start_ts to the next timestamp after the last received candle
        start_ts = int(data[-1][0]) + 1
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.1)
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                     'close_time', 'quote_volume', 'num_trades',
                     'taker_base_vol', 'taker_quote_vol', 'ignore']
        
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Set index to open_time
        df.set_index('open_time', inplace=True)
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df
    return None

def get_usdt_symbols():
    """Get all USDT trading pairs from Binance.
    
    Returns:
        list: List of trading pair symbols ending in USDT
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = json.loads(requests.get(url).text)
    symbols = [x['symbol'] for x in response['symbols'] if x['symbol'].endswith('USDT')]
    return symbols

def get_symbols_from_list(symbol_list):
    """Get trading pairs from Binance that exist in the provided list.
    
    Args:
        symbol_list (list): List of symbol names to check
        
    Returns:
        list: List of trading pair symbols that exist on Binance
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = json.loads(requests.get(url).text)
    available_symbols = [x['symbol'] for x in response['symbols']]
    return [symbol for symbol in symbol_list if symbol in available_symbols]