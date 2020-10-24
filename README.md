# bybit-backtest

[![PyPI](https://img.shields.io/pypi/v/bybit-backtest)](https://pypi.org/project/bybit-backtest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/10mohi6/bybit-backtest-python/branch/master/graph/badge.svg)](https://codecov.io/gh/10mohi6/bybit-backtest-python)
[![Build Status](https://travis-ci.com/10mohi6/bybit-backtest-python.svg?branch=master)](https://travis-ci.com/10mohi6/bybit-backtest-python)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bybit-backtest)](https://pypi.org/project/bybit-backtest/)
[![Downloads](https://pepy.tech/badge/bybit-backtest)](https://pepy.tech/project/bybit-backtest)

bybit-backtest is a python library for backtest with bybit fx trade on Python 3.6 and above.


## Installation

    $ pip install bybit-backtest

## Usage

### basic run
```python
from bybit_backtest import Backtest

class MyBacktest(Backtest):
    def strategy(self):
        fast_ma = self.sma(period=5)
        slow_ma = self.sma(period=25)
        # golden cross
        self.sell_exit = self.buy_entry = (fast_ma > slow_ma) & (
            fast_ma.shift() <= slow_ma.shift()
        )
        # dead cross
        self.buy_exit = self.sell_entry = (fast_ma < slow_ma) & (
            fast_ma.shift() >= slow_ma.shift()
        )

MyBacktest().run()
```

### advanced run
```python
from bybit_backtest import Backtest

class MyBacktest(Backtest):
    def strategy(self):
        rsi = self.rsi(period=10)
        ema = self.ema(period=20)
        lower = ema - (ema * 0.001)
        upper = ema + (ema * 0.001)
        self.buy_entry = (rsi < 30) & (self.df.C < lower)
        self.sell_entry = (rsi > 70) & (self.df.C > upper)
        self.sell_exit = ema > self.df.C
        self.buy_exit = ema < self.df.C
        self.qty = 0.1 # order quantity (default=0.001)
        self.stop_loss = 50 # stop loss (default=0 stop loss none)
        self.take_profit = 100 # take profit (default=0 take profit none)

MyBacktest(
    symbol="BTCUSD", # default=BTCUSD
    sqlite_file_name="backtest.sqlite3", # (default=backtest.sqlite3)
    from_datetime="2020-04-01", # (default="")
    to_datetime="2020-10-25", # (default="")
    interval="1T", # 5-60S(second), 1-60T(minute), 1-24H(hour) (default=1T)
    download_data_dir="data", # download data directory (default=data)
).run("backtest.png")
```
```python
total profit          491.800
total trades        10309.000
win rate               65.700
profit factor           1.047
maximum drawdown      135.500
recovery factor         3.630
riskreward ratio        0.551
sharpe ratio            0.020
average return          0.001
stop loss            1779.000
take profit            93.000
```
![backtest.png](https://raw.githubusercontent.com/10mohi6/bybit-backtest-python/master/tests/backtest.png)


## Supported indicators
- Simple Moving Average 'sma'
- Exponential Moving Average 'ema'
- Moving Average Convergence Divergence 'macd'
- Relative Strenght Index 'rsi'
- Bollinger Bands 'bbands'
- Stochastic Oscillator 'stoch'


## Getting started
For help getting started with Bybit APIs and Websocket, view our online [documentation](https://bybit-exchange.github.io/docs/inverse/#t-introduction).
