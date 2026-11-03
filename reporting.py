from cProfile import label

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def calculate_sharpe(
    equity: pd.Series,
    annual_risk_free_rate: float = 0.0,
) -> float:
    daily_equity = (
        equity.sort_index()
        .tz_convert("America/New_York")
        .resample("D")
        .last()
        .dropna()
    )

    returns = daily_equity.pct_change(fill_method=None).dropna()
    daily_risk_free_rate = annual_risk_free_rate / 252
    excess_returns = returns - daily_risk_free_rate

    volatility = excess_returns.std(ddof=1)

    if pd.isna(volatility) or volatility == 0:
        return float("nan")

    return float(
        excess_returns.mean() / volatility * (252 ** 0.5)
    )


def summarize(result, starting_cash, slippage_rate):
    equity = pd.DataFrame(result["equity_history"]).set_index("ts")["value"]
    trades = pd.DataFrame(result["trade_history"])
    final_equity = equity.iloc[-1]
    peaks = equity.cummax().clip(lower=starting_cash)
    drawdown = (1 - equity / peaks).max() * 100

    fees = 0
    slippage = 0
    win_rate = 0
    if not trades.empty:
        fees = (trades["entry_fee"] + trades["exit_fee"]).sum()
        buy_slippage = trades["entry_price"] - trades["entry_price"] / (1 + slippage_rate)
        sell_slippage = trades["exit_price"] / (1 - slippage_rate) - trades["exit_price"]
        slippage = (trades["qty"] * (buy_slippage + sell_slippage)).sum()
        profits = trades["net_profit"]
        win_rate = (profits > 0).mean() * 100

    return {
        "Final equity": final_equity,
        "Net P/L": final_equity - starting_cash,
        "Return %": (final_equity / starting_cash - 1) * 100,
        "Trades": len(trades),
        "Win rate %": win_rate,
        "Fees": fees,
        "Slippage": slippage,
        "Max drawdown %": drawdown,
        "Sharpe ratio": calculate_sharpe(equity, 0)
    }


def plot(result: dict, symbol: str, optimized_result: dict | None = None):
    eq = pd.DataFrame(result["equity_history"]).set_index("ts")
    strategies = [("Unoptimized strategy", result)]
    if optimized_result is not None:
        strategies.append(("Optimized strategy", optimized_result))

    fig, axes = plt.subplots(
        nrows=2 * len(strategies) + 1,
        ncols=1,
        sharex=True,
        figsize=(21, 3.5 * (2 * len(strategies) + 1)),
    )

    # PRICE, EMA & ENTRY/EXIT
    fig.suptitle(symbol)
    for ax, (title, strategy_result) in zip(axes, strategies):
        strategy_equity = pd.DataFrame(strategy_result["equity_history"]).set_index("ts")
        df = strategy_result["dataframe"]
        if optimized_result is not None:
            ax.set_title(title)
        ax.plot(strategy_equity.index, strategy_equity["close"], label="Close")
        ax.plot(df.index, df["ema"], label="EMA")

        trades = pd.DataFrame(strategy_result["trade_history"])
        if not trades.empty:
            ax.scatter(
                trades["entry_ts"],
                trades["entry_price"],
                color="blue",
                marker="o",
                s=75,
                zorder=10,
                label="Entry"
            )

            exit_styles = {
                "take_profit": ("green", "X"),
                "stop_loss": ("red", "X"),
                "data_end": ("black", "X"),
                "exit_signal": ("orange", "X")
            }

            for reason, (clr, marker) in exit_styles.items():
                exits = trades.loc[trades["exit_reason"] == reason]
                ax.scatter(
                    exits["exit_ts"],
                    exits["exit_price"],
                    color=clr,
                    marker=marker,
                    s=75,
                    zorder=10,
                    label=reason,
                )

        ax.set_ylabel("Close price")
        ax.legend()

    # AROON
    for ax, (title, strategy_result) in zip(axes[len(strategies):-1], strategies):
        df = strategy_result["dataframe"]
        ax.set_title(f"Aroon - {title}")
        ax.plot(df.index, df["aroon_up"], color="green", label="Aroon Up")
        ax.plot(df.index, df["aroon_down"], color="red", label="Aroon Down")
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Aroon")
        ax.legend()

    # EQUITY
    axes[-1].plot(
        eq.index,
        eq["value"],
        color="blue",
        label="Unoptimized" if optimized_result is not None else "Equity over time",
    )
    if optimized_result is not None:
        optimized_equity = (
            pd.DataFrame(optimized_result["equity_history"]).set_index("ts")
        )
        axes[-1].plot(
            optimized_equity.index,
            optimized_equity["value"],
            color="green",
            label="Optimized",
        )
    axes[-1].axhline(y=result["starting_cash"], color="gray", linestyle="--", label="Starting cash")
    axes[-1].set_ylabel("Cash + stock ($)")
    axes[-1].set_xlabel("Date")
    axes[-1].legend()

    for ax in axes:
        ax.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=15, maxticks=30)
        )
        ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    plt.savefig(f"plots/{symbol}_2024.png")
    plt.show()
