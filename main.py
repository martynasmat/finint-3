from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from backtest import Backtest
from indicators import *

STARTING_CASH = 100000
TRADING_FREE = 0.001
SLIPPAGE_RATE = 0.0005

def calculate_indicators(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"./data/{symbol}.csv",
        index_col="timestamp",
        parse_dates=["timestamp"]
    )

    df = calculate_aroon(df, 25)
    df = calculate_atr(df, 14)
    df = calculate_ema(df, 250)

    return df


def create_and_run_backtest(df: pd.DataFrame):
    return Backtest(
        df,
        STARTING_CASH,
        TRADING_FREE,
        SLIPPAGE_RATE
    ).run()

symbols = [
    "AAPL",
    "SNDK",
    "LMT",
    "INTC",
    "AMZN",
    "GOOG",
    "MSFT",
    "PLTR",
    "CSCO",
]

with ThreadPoolExecutor(max_workers=3) as executor:
    dfs = executor.map(calculate_indicators, symbols)
    res = executor.map(create_and_run_backtest, dfs)

for r in res:
    print(r)