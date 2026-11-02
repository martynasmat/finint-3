from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from backtest import Backtest
from reporting import summarize, plot

STARTING_CASH = 100000
TRADING_FEE = 0.001
SLIPPAGE_RATE = 0.0005

def create_and_run_backtest(symbol: str):
    return Backtest(
        df=pd.read_csv(
            f"data/{symbol}.csv",
            index_col="timestamp",
            parse_dates=["timestamp"]
        ).sort_index(),
        cash=STARTING_CASH,
        trading_fee=TRADING_FEE,
        slippage_rate=SLIPPAGE_RATE,
        risk_per_trade=0.1,
        risk_multiplier=4,
        aroon_up_entry_signal=70,
        aroon_down_entry_signal=30,
        ema_period=200,
        aroon_period=25,
        atr_period=14,
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
        results = list(executor.map(create_and_run_backtest, symbols))

    summaries = {}
    for symbol, result in zip(symbols, results):
        stats = summarize(result, STARTING_CASH, SLIPPAGE_RATE)
        plot(result, symbol)
        summaries[symbol] = stats

    summary = pd.DataFrame.from_dict(summaries, orient="index").rename_axis("Symbol")
    print(f"\nStarting cash ${STARTING_CASH:,.2f}")
    print(f"Trading fee: {TRADING_FEE:.2%}; slippage: {SLIPPAGE_RATE:.2%}")
    print(summary.to_string(float_format=lambda value: f"{value:,.2f}"))


if __name__ == "__main__":
    main()
