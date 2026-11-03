from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import pandas as pd
from backtest import Backtest
from reporting import summarize, plot

STARTING_CASH = 100000
TRADING_FEE = 0.001
SLIPPAGE_RATE = 0.0005

def create_and_run_backtest(df: pd.DataFrame, params: dict):
    return Backtest(
        df=df.copy(),
        cash=STARTING_CASH,
        trading_fee=TRADING_FEE,
        slippage_rate=SLIPPAGE_RATE,
        **params,
    ).run()


symbols = [
    "AAPL",
    "LMT",
    "INTC",
    "AMZN",
    "GOOG",
    "MSFT",
    "PLTR",
    "CSCO",
]

def main():
    dfs = [
        pd.read_csv(
            f"data/{symbol}.csv",
            index_col="timestamp",
            parse_dates=["timestamp"],
        ).sort_index()
        for symbol in symbols
    ]

    unoptimized_params = {
        "risk_per_trade": 0.3,
        "stop_loss_multiplier": 4,
        "take_profit_multiplier": 2,
        "aroon_up_entry_signal": 70,
        "aroon_down_entry_signal": 30,
        "ema_period": 200,
        "aroon_period": 25,
        "atr_period": 14,
    }

    optimized_params = {
        "risk_per_trade": 0.3,
        "stop_loss_multiplier": 4,
        "take_profit_multiplier": 2,
        "aroon_up_entry_signal": 70,
        "aroon_down_entry_signal": 50,
        "ema_period": 1200,
        "aroon_period": 200,
        "atr_period": 14,
    }

    with ThreadPoolExecutor(max_workers=10) as executor:
        unoptimized_results = list(executor.map(
            create_and_run_backtest,
            dfs,
            repeat(unoptimized_params),
        ))
        optimized_results = list(executor.map(
            create_and_run_backtest,
            dfs,
            repeat(optimized_params),
        ))

    summaries = []
    for symbol, unoptimized_result, optimized_result in zip(
        symbols, unoptimized_results, optimized_results
    ):
        for label, result in (
            ("Unoptimized", unoptimized_result),
            ("Optimized", optimized_result),
        ):
            summaries.append({
                "Symbol": symbol,
                "Parameters": label,
                **summarize(result, STARTING_CASH, SLIPPAGE_RATE),
            })

        plot(unoptimized_result, symbol, optimized_result=optimized_result)

    summary = pd.DataFrame(summaries).set_index(["Symbol", "Parameters"])
    print(summary.to_string(float_format=lambda value: f"{value:,.2f}"))


if __name__ == "__main__":
    main()
