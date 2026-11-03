from collections.abc import Callable
from itertools import product

import pandas as pd

from backtest import Backtest
from reporting import summarize


def optimize(
    dfs: list[pd.DataFrame],
    parameter_grid: dict[str, list],
    fixed_params: dict,
    objective: Callable[[dict], float],
) -> pd.DataFrame:
    parameter_names = list(parameter_grid.keys())
    parameter_options = list(parameter_grid.values())
    results = []

    for values in product(*parameter_options):
        candidate = {}
        for name, value in zip(parameter_names, values):
            candidate[name] = value

        params = fixed_params.copy()
        params.update(candidate)

        total_score = 0.0
        for df in dfs:
            backtest = Backtest(df=df.copy(), **params)
            result = backtest.run()
            total_score += float(objective(result))

        candidate["average_score"] = total_score / len(dfs)
        results.append(candidate)

    rankings = pd.DataFrame(results)
    rankings = rankings.sort_values(
        "average_score", ascending=False, kind="stable"
    )
    return rankings.reset_index(drop=True)


def main():
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
    dfs = [
        pd.read_csv(
            f"data/{symbol}.csv",
            index_col="timestamp",
            parse_dates=["timestamp"],
        ).sort_index()
        for symbol in symbols
    ]

    parameter_grid = {
        "aroon_period": [14, 32, 100, 200],
        "ema_period": [50, 250, 400, 1200],
        "stop_loss_multiplier": [2, 4, 8],
        "aroon_up_entry_signal": [60, 70, 90],
        "aroon_down_entry_signal": [10, 30, 50],
        "atr_period": [7, 14],
    }

    fixed_params = {
        "cash": 100000,
        "trading_fee": 0.001,
        "slippage_rate": 0.0005,
        "take_profit_multiplier": 2,
        "risk_per_trade": 0.3,
    }

    rankings = optimize(
        dfs=dfs,
        parameter_grid=parameter_grid,
        fixed_params=fixed_params,
        objective=lambda result: summarize(
            result, fixed_params["cash"], fixed_params["slippage_rate"]
        )["Sharpe ratio"],
    )
    print(
        rankings.head(5)
        .rename(columns={"average_score": "Sharpe ratio"})
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
