import pandas as pd

def calculate_ma(df):

    df['MA_20'] = df['Close'].rolling(window=20).mean()

    return df