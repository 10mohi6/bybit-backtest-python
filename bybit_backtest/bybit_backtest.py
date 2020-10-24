from typing import Tuple
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
import urllib.request
import gzip
import datetime


class Backtest(object):
    def __init__(
        self,
        *,
        symbol: str = "BTCUSD",
        sqlite_file_name: str = "backtest.sqlite3",
        from_datetime: str = "",
        to_datetime: str = "",
        qty: float = 0.001,
        interval: str = "1T",  # 5-60S(second), 1-60T(minute), 1-24H(hour)
        download_data_dir: str = "data",
    ) -> None:
        self.interval = interval
        self.download_data_dir = download_data_dir
        self.qty = qty
        self.take_profit = 0
        self.stop_loss = 0
        self.buy_entry = (
            self.buy_exit
        ) = self.sell_entry = self.sell_exit = pd.DataFrame()
        self.symbol = symbol
        self.engine = create_engine(f"sqlite:///{sqlite_file_name}")
        self.from_datetime = from_datetime
        self.to_datetime = to_datetime
        if not os.path.exists(self.download_data_dir):
            os.mkdir(self.download_data_dir)

    def strategy(self) -> None:
        pass

    def _download_data_to_db(self) -> None:
        data_url = f"https://public.bybit.com/trading/{self.symbol}/"
        res = requests.get(data_url)
        if res.status_code != 200:
            raise Exception(f"requests error {res.status_code} {res.json()}")
        interval = "5S"
        soup = BeautifulSoup(res.text, "lxml")
        for link in soup.find_all("a"):
            name = link.get("href")
            if (
                self.from_datetime != ""
                and f"{self.symbol}{self.from_datetime}.csv.gz" > name
            ):
                continue
            if (
                self.to_datetime != ""
                and f"{self.symbol}{self.to_datetime}.csv.gz" < name
            ):
                continue
            fname = f"{self.download_data_dir}/{name}"
            if os.path.exists(fname):
                continue
            urllib.request.urlretrieve(f"{data_url}{name}", fname)
            print(f"download {name}")
            json = []
            with gzip.open(fname, mode="rt") as fp:
                for index, line in enumerate(fp.read().split("\n")):
                    ary = line.split(",")
                    if index == 0:
                        continue
                    if len(line) == 0:
                        continue
                    json.append(
                        {
                            "time": datetime.datetime.fromtimestamp(
                                int(float(ary[0])), datetime.timezone.utc
                            ),
                            "size": float(ary[3]),
                            "price": float(ary[4]),
                        }
                    )
            df = pd.DataFrame(json)
            df.index = pd.DatetimeIndex(df["time"])
            df.index.names = ["T"]
            df = (
                df.resample(interval).agg(
                    {
                        "price": "ohlc",
                        "size": "sum",
                    }
                )
            ).fillna(method="bfill")
            df.columns = ["O", "H", "L", "C", "V"]
            df.index.astype(str)
            df.to_sql(self.symbol, con=self.engine, if_exists="append", index_label="T")

    def _create_candles(self) -> None:
        sql = ""
        where = []
        if self.from_datetime != "":
            where.append(f" T >= '{self.from_datetime} 00:00:00' ")
        if self.to_datetime != "":
            where.append(f" T <= '{self.to_datetime} 00:00:00' ")
        if len(where) > 0:
            sql = f"WHERE{'and'.join(where)}"
        self.df = pd.read_sql_query(
            f"SELECT * FROM {self.symbol} {sql} ORDER BY T",
            self.engine,
            index_col="T",
        )
        self.df.index = pd.to_datetime(self.df.index)
        if self.interval != "5S":
            self.df = (
                self.df.resample(self.interval).agg(
                    {
                        "O": "first",
                        "H": "max",
                        "L": "min",
                        "C": "last",
                        "V": "sum",
                    }
                )
            ).fillna(method="bfill")

    def run(self, filename: str = "") -> None:
        self._download_data_to_db()
        self._create_candles()
        self.strategy()
        o = self.df.O.values
        L = self.df.L.values
        h = self.df.H.values
        N = len(self.df)
        long_trade = np.zeros(N)
        short_trade = np.zeros(N)

        # buy entry
        buy_entry_s = np.hstack((False, self.buy_entry[:-1]))  # shift
        long_trade[buy_entry_s] = o[buy_entry_s]
        # buy exit
        buy_exit_s = np.hstack((False, self.buy_exit[:-2], True))  # shift
        long_trade[buy_exit_s] = -o[buy_exit_s]
        # sell entry
        sell_entry_s = np.hstack((False, self.sell_entry[:-1]))  # shift
        short_trade[sell_entry_s] = o[sell_entry_s]
        # sell exit
        sell_exit_s = np.hstack((False, self.sell_exit[:-2], True))  # shift
        short_trade[sell_exit_s] = -o[sell_exit_s]

        long_pl = pd.Series(np.zeros(N))  # profit/loss of buy position
        short_pl = pd.Series(np.zeros(N))  # profit/loss of sell position
        buy_price = sell_price = 0
        long_rr = []  # long return rate
        short_rr = []  # short return rate
        stop_loss = take_profit = 0

        for i in range(1, N):
            # buy entry
            if long_trade[i] > 0:
                if buy_price == 0:
                    buy_price = long_trade[i]
                    short_trade[i] = -buy_price  # sell exit
                else:
                    long_trade[i] = 0

            # sell entry
            if short_trade[i] > 0:
                if sell_price == 0:
                    sell_price = short_trade[i]
                    long_trade[i] = -sell_price  # buy exit
                else:
                    short_trade[i] = 0

            # buy exit
            if long_trade[i] < 0:
                if buy_price != 0:
                    long_pl[i] = (
                        -(buy_price + long_trade[i]) * self.qty
                    )  # profit/loss fixed
                    long_rr.append(
                        round(long_pl[i] / buy_price * 100, 2)
                    )  # long return rate
                    buy_price = 0
                else:
                    long_trade[i] = 0

            # sell exit
            if short_trade[i] < 0:
                if sell_price != 0:
                    short_pl[i] = (
                        sell_price + short_trade[i]
                    ) * self.qty  # profit/loss fixed
                    short_rr.append(
                        round(short_pl[i] / sell_price * 100, 2)
                    )  # short return rate
                    sell_price = 0
                else:
                    short_trade[i] = 0

            # close buy position with stop loss
            if buy_price != 0 and self.stop_loss > 0:
                stop_price = buy_price - self.stop_loss
                if L[i] <= stop_price:
                    long_trade[i] = -stop_price
                    long_pl[i] = (
                        -(buy_price + long_trade[i]) * self.qty
                    )  # profit/loss fixed
                    long_rr.append(
                        round(long_pl[i] / buy_price * 100, 2)
                    )  # long return rate
                    buy_price = 0
                    stop_loss += 1

            # close buy positon with take profit
            if buy_price != 0 and self.take_profit > 0:
                limit_price = buy_price + self.take_profit
                if h[i] >= limit_price:
                    long_trade[i] = -limit_price
                    long_pl[i] = (
                        -(buy_price + long_trade[i]) * self.qty
                    )  # profit/loss fixed
                    long_rr.append(
                        round(long_pl[i] / buy_price * 100, 2)
                    )  # long return rate
                    buy_price = 0
                    take_profit += 1

            # close sell position with stop loss
            if sell_price != 0 and self.stop_loss > 0:
                stop_price = sell_price + self.stop_loss
                if h[i] >= stop_price:
                    short_trade[i] = -stop_price
                    short_pl[i] = (
                        sell_price + short_trade[i]
                    ) * self.qty  # profit/loss fixed
                    short_rr.append(
                        round(short_pl[i] / sell_price * 100, 2)
                    )  # short return rate
                    sell_price = 0
                    stop_loss += 1

            # close sell position with take profit
            if sell_price != 0 and self.take_profit > 0:
                limit_price = sell_price - self.take_profit
                if L[i] <= limit_price:
                    short_trade[i] = -limit_price
                    short_pl[i] = (
                        sell_price + short_trade[i]
                    ) * self.qty  # profit/loss fixed
                    short_rr.append(
                        round(short_pl[i] / sell_price * 100, 2)
                    )  # short return rate
                    sell_price = 0
                    take_profit += 1

        win_trades = np.count_nonzero(long_pl.clip(lower=0)) + np.count_nonzero(
            short_pl.clip(lower=0)
        )
        lose_trades = np.count_nonzero(long_pl.clip(upper=0)) + np.count_nonzero(
            short_pl.clip(upper=0)
        )
        trades = (np.count_nonzero(long_trade) // 2) + (
            np.count_nonzero(short_trade) // 2
        )
        gross_profit = long_pl.clip(lower=0).sum() + short_pl.clip(lower=0).sum()
        gross_loss = long_pl.clip(upper=0).sum() + short_pl.clip(upper=0).sum()
        profit_pl = gross_profit + gross_loss
        self.equity = (long_pl + short_pl).cumsum()
        mdd = (self.equity.cummax() - self.equity).max()
        self.return_rate = pd.Series(short_rr + long_rr)

        s = pd.Series(dtype="object")
        s.loc["total profit"] = round(profit_pl, 3)
        s.loc["total trades"] = trades
        s.loc["win rate"] = round(win_trades / trades * 100, 3)
        s.loc["profit factor"] = round(-gross_profit / gross_loss, 3)
        s.loc["maximum drawdown"] = round(mdd, 3)
        s.loc["recovery factor"] = round(profit_pl / mdd, 3)
        s.loc["riskreward ratio"] = round(
            -(gross_profit / win_trades) / (gross_loss / lose_trades), 3
        )
        s.loc["sharpe ratio"] = round(
            self.return_rate.mean() / self.return_rate.std(), 3
        )
        s.loc["average return"] = round(self.return_rate.mean(), 3)
        s.loc["stop loss"] = stop_loss
        s.loc["take profit"] = take_profit
        print(s)

        fig = plt.figure(figsize=(8, 4))
        fig.subplots_adjust(
            wspace=0.2, hspace=0.5, left=0.095, right=0.95, bottom=0.095, top=0.95
        )
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(self.df.C, label="close")
        ax1.legend()
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(self.equity, label="equity")
        ax2.legend()
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.hist(self.return_rate, 50, rwidth=0.9)
        ax3.axvline(
            sum(self.return_rate) / len(self.return_rate),
            color="orange",
            label="average return",
        )
        ax3.legend()
        if filename == "":
            plt.show()
        else:
            plt.savefig(filename)

    def sma(self, *, period: int, price: str = "C") -> pd.DataFrame:
        return self.df[price].rolling(period).mean()

    def ema(self, *, period: int, price: str = "C") -> pd.DataFrame:
        return self.df[price].ewm(span=period).mean()

    def bbands(
        self, *, period: int = 20, band: int = 2, price: str = "C"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        std = self.df[price].rolling(period).std()
        mean = self.df[price].rolling(period).mean()
        return mean + (std * band), mean, mean - (std * band)

    def macd(
        self,
        *,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price: str = "C",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        macd = (
            self.df[price].ewm(span=fast_period).mean()
            - self.df[price].ewm(span=slow_period).mean()
        )
        signal = macd.ewm(span=signal_period).mean()
        return macd, signal

    def stoch(
        self, *, k_period: int = 5, d_period: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        k = (
            (self.df.C - self.df.L.rolling(k_period).min())
            / (self.df.H.rolling(k_period).max() - self.df.L.rolling(k_period).min())
            * 100
        )
        d = k.rolling(d_period).mean()
        return k, d

    def rsi(self, *, period: int = 14, price: str = "C") -> pd.DataFrame:
        return 100 - 100 / (
            1
            - self.df[price].diff().clip(lower=0).rolling(period).mean()
            / self.df[price].diff().clip(upper=0).rolling(period).mean()
        )
