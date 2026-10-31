import pandas as pd
import numpy as np

def calculate_aroon(df: pd.DataFrame, period: int=25) -> pd.DataFrame:
    res = df.copy()
    window = period + 1

    high_age = res["high"].rolling(
        window, min_periods=window
    ).apply(lambda x: np.argmax(x[::-1]), raw=True)

    low_age = res["low"].rolling(
        window, min_periods=window
    ).apply(lambda x: np.argmin(x[::-1]), raw=True)

    res["aroon_up"] = 100 * (period - high_age) / period
    res["aroon_down"] = 100 * (period - low_age) / period

    return res