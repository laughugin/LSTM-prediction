# Import necessary libraries
from binance.client import Client
import configparser
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class HistoricalData:

    # Parameters for request
    interval = "1m"
    limit = 1000

    def __init__(self, base, quote, start_time, end_time) -> None:

        # Currency to take info from
        self.base = "BTC"
        self.quote = "USDT"
        # For request
        self.symbol = self.base + self.quote

        self.start_time = start_time
        self.end_time = end_time

        # Read API configuration
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Log in with API credentials
        self.__api_key = config['API']['key']
        self.__api_secret = config['API']['secret_key']
        self.client = Client(self.__api_key, self.__api_secret)

    def get_historical_data(self, start_time_ms, end_time_ms):
        kline = self.client.get_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=self.limit,
            startTime=start_time_ms,
            endTime=end_time_ms
        )
        data = []
        for price in kline:
            timestamp = datetime.utcfromtimestamp(price[0] / 1000)
            close_price = price[4]
            data.append({'timestamp': timestamp, 'close_price': close_price})
        return data

    def get_data(self):
        df = pd.DataFrame(columns=['timestamp', 'close_price'])

        while True:
            if not df.empty:
                last_timestamp = df['timestamp'].iloc[-1]
                start = last_timestamp + timedelta(seconds=1)
            else:
                start = self.start_time

            end = self.end_time

            # Use local variables for milliseconds conversion
            start_time_ms = int(start.timestamp() * 1000)
            end_time_ms = int(end.timestamp() * 1000)

            # Fetch data from Binance API
            data = self.get_historical_data(start_time_ms, end_time_ms)  # Pass the millisecond timestamps to the method

            # Append the new data to the DataFrame
            df = df._append(data, ignore_index=True)  # Use 'append' instead of '_append'

            if df['timestamp'].iloc[-1] >= end - timedelta(hours=1, minutes=1):
                break

        df.to_csv(self.base + self.quote + ".csv", index=False)





