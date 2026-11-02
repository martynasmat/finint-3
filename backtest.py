import pandas as pd

class Backtest:

    def __init__(
            self,
            df: pd.DataFrame,
            cash: float,
            trading_fee: float,
            slippage_rate: float,
    ):
        self.df = df
        self.cash = cash
        self.trading_fee = trading_fee
        self.slippage_rate = slippage_rate
        self.qty = 0
        self.stop_price = 0
        self.target_price = 0
        self.entry_time = None
        self.entry_price = None
        self.entry_fee = 0.0

        self.trades = []

    def set_entry_signals(self):
        condition = (
            (self.df["aroon_up"] >= 70)
            & (self.df["aroon_down"] <= 30)
            & (self.df["close"] > self.df["ema"])
        )

        self.df["entry"] = condition & ~condition.shift(1, fill_value=False)

    def open_position(self, ts: pd.Timestamp, price: float, atr: float) -> None:
        if self.qty > 0:
            return

        fill_price = price * (1 + self.slippage_rate)
        cost_per_share = fill_price * (1 + self.trading_fee)
        qty = min(10, int(self.cash // cost_per_share))
        if qty <= 0:
            return

        purchase_value = qty * fill_price
        entry_fee = purchase_value * self.trading_fee
        total_cost = purchase_value + entry_fee
        risk_per_share = atr * 2

        self.cash -= total_cost
        self.qty = qty
        self.entry_time = ts
        self.entry_price = fill_price
        self.entry_fee = entry_fee
        self.stop_price = fill_price - risk_per_share
        self.target_price = fill_price + risk_per_share * 2

    def close_position(self, ts: pd.Timestamp, price: float, reason: str) -> None:
        if self.qty == 0:
            return

        exit_price = price * (1 - self.slippage_rate)
        sale_value = exit_price * self.qty
        exit_fee = sale_value * self.trading_fee

        self.cash += sale_value - exit_fee

        net_profit = self.qty * (exit_price - self.entry_price) - self.entry_fee - exit_fee
        self.trades.append(
            {
                "entry_ts": self.entry_time,
                "exit_ts": ts,
                "entry_price": self.entry_price,
                "exit_price": exit_price,
                "entry_fee": self.entry_fee,
                "exit_fee": exit_fee,
                "qty": self.qty,
                "net_profit": net_profit,
                "exit_reason": reason,
            }
        )

        self.qty = 0
        self.entry_time = None
        self.entry_price = None
        self.entry_fee = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0

    def run(self):
        self.set_entry_signals()
        prev = None

        for curr in self.df.itertuples():
            if prev is not None:
                if (
                    self.qty == 0
                    and prev.entry
                    and pd.notna(prev.atr)
                    and prev.atr > 0
                ):
                    self.open_position(curr.Index, curr.open, prev.atr)

                if self.qty > 0:
                    if curr.open <= self.stop_price:
                        self.close_position(curr.Index, curr.open, "stop_loss")
                    elif curr.open >= self.target_price:
                        self.close_position(curr.Index, curr.open, "take_profit")
                    elif curr.low <= self.stop_price:
                        self.close_position(curr.Index, self.stop_price, "stop_loss")
                    elif curr.high >= self.target_price:
                        self.close_position(curr.Index, self.target_price, "take_profit")

            equity = self.cash + self.qty * curr.close
            prev = curr
