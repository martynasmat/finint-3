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


def calculate_ema(df: pd.DataFrame, period: int=250) -> pd.DataFrame:
    res = df.copy()
    alpha = 2 / (period + 1)

    res["ema"] = res["close"].ewm(
        alpha=alpha,
        min_periods=period,
        adjust=False,
    ).mean()

    return res


def calculate_atr(df: pd.DataFrame, period: int=14) -> pd.DataFrame:
    res = df.copy()
    alpha = 1 / period
    prev_close = res["close"].shift(1)
    tr = pd.concat(
        [
            res["high"] - res["low"],
            abs(res["high"] - prev_close),
            abs(res["low"] - prev_close),
        ],
        axis=1,
    ).max(axis=1)

    res["atr"] = tr.ewm(
        alpha=alpha,
        min_periods=period,
        adjust=False,
    ).mean()

    return res
