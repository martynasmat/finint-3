import datetime
import alpaca.data
from alpaca.data import StockHistoricalDataClient, StockTradesRequest, StockLatestTradeRequest, StockBarsRequest, \
TimeFrame

class DataClient:
    def __init__(self, api_key, secret_key):
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def fetch_tick_data(
            self,
            symbol: str,
            start_date: datetime.date,
            end_date: datetime.date,
            t: int = 100,
        ) -> alpaca.data.TradeSet:
        """
        Returns tick data for a given stock ticker.
        :param symbol: Stock ticker
        :param start_date: Start date of data
        :param end_date: End date of data
        :param t: Limit of trades to return
        :return: TradeSet object
        """
        req = StockTradesRequest(
        symbol_or_symbols=symbol,
        limit=t,
        start=start_date,
        end=end_date,
        )
        return self.client.get_stock_trades(
            request_params=req,
        )

    def fetch_bar_data(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
        limit: int = 100,
        tf: TimeFrame = TimeFrame.Day,
    ) -> alpaca.data.BarSet:
        """
        Returns bar data for a given stock ticker.
        :param symbol: Stock ticker
        :param start_date: Start date of data
        :param end_date: End date of data
        :param limit: Limit of bars to return
        :param tf: Timeframe for bar (day/minute)
        :return: BarSet object
        """
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            limit=limit,
            start=start_date,
            end=end_date,
        )
        return self.client.get_stock_bars(request_params=req,)
