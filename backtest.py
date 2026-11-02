import pandas as pd
import numpy as np

class Backtest:
    def __init__(
            self,
            df: pd.DataFrame,
            cash: float,
            trading_fee: float,
            slippage_rate: float,
            risk_per_trade: float = 0.1,
            risk_multiplier: float = 4,
            aroon_up_entry_signal: int = 70,
            aroon_down_entry_signal: int = 30,
            ema_period: int = 200,
            aroon_period: int = 25,
            atr_period: int = 14,
    ):
        self.df = df
        self.cash = self.starting_cash = cash
        self.trading_fee = trading_fee
        self.slippage_rate = slippage_rate

        # Percentage of total bankroll per trade
        self.risk_per_trade = risk_per_trade
        # Multiplier used to determine stop-loss and take-profit prices (ATR * risk_multiplier)
        self.risk_multiplier = risk_multiplier
        # Aroon Up/Down values used for entry signals
        self.aroon_up_entry_signal = aroon_up_entry_signal
        self.aroon_down_entry_signal = aroon_down_entry_signal

        self.ema_period = ema_period
        self.aroon_period = aroon_period
        self.atr_period = atr_period

        self.qty = 0
        self.stop_price = 0
        self.target_price = 0
        self.entry_time = None
        self.entry_price = None
        self.entry_fee = 0.0

        self.trades = []
        self.equity = []

    def set_entry_signals(self):
        condition = (
            (self.df["aroon_up"] >= 70)
            & (self.df["aroon_down"] <= 30)
            & (self.df["close"] > self.df["ema"])
        )

        self.df["entry"] = condition & ~condition.shift(1, fill_value=False)

    def set_exit_signals(self):
        condition = (self.df["aroon_up"] < self.df["aroon_down"]) & (self.df["close"] < self.df["ema"])

        self.df["exit"] = condition & ~condition.shift(1, fill_value=False)

    def open_position(self, ts: pd.Timestamp, price: float, atr: float) -> None:
        if self.qty > 0:
            return

        fill_price = price * (1 + self.slippage_rate)
        cost_per_share = fill_price * (1 + self.trading_fee)
        qty = self.cash * self.risk_per_trade // cost_per_share
        if qty <= 0:
            return

        purchase_value = qty * fill_price
        entry_fee = purchase_value * self.trading_fee
        total_cost = purchase_value + entry_fee
        risk_per_share = atr * self.risk_multiplier

        self.cash -= total_cost
        self.qty = qty
        self.entry_time = ts
        self.entry_price = fill_price
        self.entry_fee = entry_fee
        self.stop_price = fill_price - risk_per_share
        self.target_price = fill_price + risk_per_share * 2

    def reset_attrs(self):
        self.qty = 0
        self.stop_price = 0
        self.target_price = 0
        self.entry_time = None
        self.entry_price = None
        self.entry_fee = 0.0

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

        self.reset_attrs()

    def calculate_ema(self):
        res = self.df.copy()
        alpha = 2 / (self.ema_period + 1)

        res["ema"] = res["close"].ewm(
            alpha=alpha,
            min_periods=self.ema_period,
            adjust=False,
        ).mean()

        self.df = res

    def calculate_aroon(self):
        res = self.df.copy()
        window = self.aroon_period + 1

        high_age = res["high"].rolling(
            window, min_periods=window
        ).apply(lambda x: np.argmax(x[::-1]), raw=True)

        low_age = res["low"].rolling(
            window, min_periods=window
        ).apply(lambda x: np.argmin(x[::-1]), raw=True)

        res["aroon_up"] = 100 * (self.aroon_period - high_age) / self.aroon_period
        res["aroon_down"] = 100 * (self.aroon_period - low_age) / self.aroon_period

        self.df = res

    def calculate_atr(self):
        res = self.df.copy()
        alpha = 1 / self.atr_period
        prev_close = res["close"].shift(1)
        tr = pd.concat(
            [
                res["high"] - res["low"],
                abs(res["high"] - prev_close),
                abs(res["low"] - prev_close),
            ],
            axis=1,
        ).max(axis=1)

        res["atr"] = tr.ewm(
            alpha=alpha,
            min_periods=self.atr_period,
            adjust=False,
        ).mean()

        self.df = res

    def calculate_indicators(self):
        self.calculate_ema()
        self.calculate_aroon()
        self.calculate_atr()

    def run(self):
        self.calculate_indicators()
        self.set_entry_signals()
        self.set_exit_signals()
        prev = None

        for curr in self.df.itertuples():
            if prev is not None:
                if self.qty == 0 and prev.entry and pd.notna(prev.atr) and prev.atr > 0:
                    self.open_position(curr.Index, curr.open, prev.atr)

                if self.qty > 0:
                    if curr.open <= self.stop_price:
                        self.close_position(curr.Index, curr.open, "stop_loss")
                    elif curr.open >= self.target_price:
                        self.close_position(curr.Index, curr.open, "take_profit")
                    elif prev.exit:
                        self.close_position(curr.Index, curr.open, "exit_signal")
                    elif curr.low <= self.stop_price:
                        self.close_position(curr.Index, self.stop_price, "stop_loss")
                    elif curr.high >= self.target_price:
                        self.close_position(curr.Index, self.target_price, "take_profit")


            self.equity.append({
                "ts": curr.Index,
                "value": self.cash + self.qty * curr.close,
                "close": curr.close,
            })
            prev = curr

        if self.qty > 0:
            self.close_position(curr.Index, curr.close, "data_end")
            self.equity[-1]["value"] = self.cash

        return {
            "equity_history": self.equity,
            "trade_history": self.trades,
            "dataframe": self.df,
            "starting_cash": self.starting_cash,
        }