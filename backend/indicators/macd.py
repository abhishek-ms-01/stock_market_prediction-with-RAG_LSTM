import pandas as pd
from ta.trend import MACD

def calculate_macd(df):

    macd = MACD(close=df['Close'])

    df['MACD'] = macd.macd()

    return df