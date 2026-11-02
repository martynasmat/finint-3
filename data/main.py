import datetime
import os
import dotenv
from data_client import DataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

def main():
    dotenv.load_dotenv()
    client = DataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY")
    )
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

    for symbol in symbols:
        res = client.fetch_bar_data(
            symbol,
            datetime.datetime(2025, 10, 1),
            datetime.datetime(2026, 10, 1),
            tf=TimeFrame(1, TimeFrameUnit.Hour),
            limit=10000000,
        )
        res.df.to_csv(f"{symbol}.csv")


if __name__ == "__main__":
    main()