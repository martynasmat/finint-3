from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from indicators import *

def calculate_indicators(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"./data/{symbol}.csv",
        index_col="timestamp",
    )

    df = calculate_aroon(df, 25)
    df = calculate_atr(df, 14)
    df = calculate_ema(df, 250)

    return df


symbols = [
    "AAPL",
    "SNDK",
    "LMT",
]

with ThreadPoolExecutor(max_workers=3) as executor:
    dfs = executor.map(calculate_indicators, symbols)

