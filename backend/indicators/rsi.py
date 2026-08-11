import pandas as pd

from ta.momentum import RSIIndicator

def calculate_rsi(df):

    rsi = RSIIndicator(close=df['Close'])

    df['RSI'] = rsi.rsi()

    return df