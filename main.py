from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from backtest import Backtest
from indicators import calculate_aroon, calculate_atr, calculate_ema
from reporting import summarize, plot

STARTING_CASH = 100000
TRADING_FEE = 0.001
SLIPPAGE_RATE = 0.0005

def calculate_indicators(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"data/{symbol}.csv",
        index_col="timestamp",
        parse_dates=["timestamp"]
    ).sort_index()

    df = calculate_aroon(df, 14)
    df = calculate_atr(df, 14)
    df = calculate_ema(df, 100)

    return df


def create_and_run_backtest(df: pd.DataFrame):
    return Backtest(
        df,
        STARTING_CASH,
        TRADING_FEE,
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

def main():
    with ThreadPoolExecutor(max_workers=10) as executor:
        dfs = list(executor.map(calculate_indicators, symbols))
        results = list(executor.map(create_and_run_backtest, dfs))

    summaries = {}
    for symbol, df, result in zip(symbols, dfs, results):
        stats = summarize(result, STARTING_CASH, SLIPPAGE_RATE)
        plot(result, symbol)
        summaries[symbol] = stats

    summary = pd.DataFrame.from_dict(summaries, orient="index").rename_axis("Symbol")
    print(f"\nStarting cash ${STARTING_CASH:,.2f}")
    print(f"Trading fee: {TRADING_FEE:.2%}; slippage: {SLIPPAGE_RATE:.2%}")
    print(summary.to_string(float_format=lambda value: f"{value:,.2f}"))


if __name__ == "__main__":
    main()
