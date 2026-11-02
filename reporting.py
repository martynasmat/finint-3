from cProfile import label

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


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
    }


def plot(result: dict, symbol: str):
    eq = pd.DataFrame(result["equity_history"]).set_index("ts")
    df = result["dataframe"]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(21, 7),
    )

    # PRICE, EMA & ENTRY/EXIT
    fig.suptitle(symbol)
    axes[0].plot(eq.index, eq["close"], label="Close")
    axes[0].plot(df.index, df["ema"], label="EMA")

    trades = pd.DataFrame(result["trade_history"])
    if not trades.empty:
        axes[0].scatter(
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
            axes[0].scatter(
                exits["exit_ts"],
                exits["entry_price"],
                color=clr,
                marker=marker,
                s=75,
                zorder=10,
                label=reason,
            )

    axes[0].set_ylabel("Close price")
    axes[0].legend()

    # AROON
    # axes[1].plot(df.index, df["aroon_up"], label="Aroon Up", color="green")
    # axes[1].plot(df.index, df["aroon_down"], label="Aroon Down", color="red")
    # axes[1].set_ylabel("Aroon Up/Down")
    # axes[1].legend()

    # EQUITY
    axes[1].plot(eq.index, eq["value"], label="Equity over time")
    axes[1].axhline(y=result["starting_cash"], color="gray", linestyle="--", label="Starting cash")
    axes[1].set_ylabel("Cash + stock ($)")
    axes[1].legend()

    for ax in axes:
        ax.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=15, maxticks=30)
        )
        ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    plt.show()
