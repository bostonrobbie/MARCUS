import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict

class Plotter:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.equity_curve = pd.DataFrame(self.portfolio.equity_curve)
        if not self.equity_curve.empty:
            self.equity_curve['datetime'] = pd.to_datetime(self.equity_curve['datetime'])
            self.equity_curve.set_index('datetime', inplace=True)

    def plot_equity_curve(self, title: str = "Strategy Equity Curve"):
        """
        Plots the equity curve (Cash + Market Value) over time.
        """
        if self.equity_curve.empty:
            print("No data to plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve['equity'], label='Equity')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_signals(self, symbol: str):
        """
        Plots the price of a symbol and overlays Buy/Sell signals.
        Requires access to the DataHandler through the Portfolio.
        """
        bars = self.portfolio.bars.symbol_data.get(symbol)
        if bars is None:
            print(f"No data found for {symbol}")
            return

        # Find signals from the portfolio history (requires tracking trade executions)
        # Currently we only have equity history. We need to look at FillEvents or infer from positions.
        # For V1 visualization, we will just plot the Equity Curve. 
        # To plot signals properly, we should have logged fills in the Portfolio.
        
        # NOTE: Portfolio currently doesn't store a list of Fills permanently.
        # We need to upgrade Portfolio to store self.all_fills or similar for this to work accurately.
        pass
