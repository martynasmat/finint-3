import pandas as pd

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