import pytest
from bybit_backtest import Backtest


@pytest.fixture(scope="module", autouse=True)
def scope_module():
    class MyBacktest(Backtest):
        def strategy(self):
            fast_ma = self.ema(period=3)
            slow_ma = self.ema(period=5)
            self.sell_exit = self.buy_entry = (fast_ma > slow_ma) & (
                fast_ma.shift() <= slow_ma.shift()
            )
            self.buy_exit = self.sell_entry = (fast_ma < slow_ma) & (
                fast_ma.shift() >= slow_ma.shift()
            )
            self.qty = 0.1
            self.stop_loss = 5
            self.take_profit = 10

    yield MyBacktest(
        symbol="BTCUSD",
        sqlite_file_name="backtest.sqlite3",
        from_date="2020-10-01",
        to_date="2020-10-10",
        interval="1T",
        download_data_dir="data",
    )


@pytest.fixture(scope="function", autouse=True)
def backtest(scope_module):
    yield scope_module


# @pytest.mark.skip
def test_backtest(backtest):
    backtest.run("backtest.png")
