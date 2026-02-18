
import unittest
import sys
import os
from collections import deque

# Path Setup
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))

from backtesting.portfolio import Portfolio, Position, OrderSide, FillEvent
from backtesting.schema import OrderSide

class TestShortPnL(unittest.TestCase):
    def test_short_round_trip(self):
        """
        Test Short Sell -> Cover Round Trip PnL.
        Short 1 @ 100. Cover 1 @ 90. Profit should be 10.
        """
        # Mock Events Queue
        events = deque()
        
        # Init Portfolio
        class MockData:
            symbol_list = ['NQ']
            def get_latest_bar(self, s): return None
            
        p = Portfolio(bars=MockData(), events=events, initial_capital=100000)
        p.instrument_config = {'NQ': {'multiplier': 1.0}} # Simple multiplier
        
        # 1. Short Sell 1 NQ @ 100
        # FillEvent: quantity is signed? No, usually unsigned + Side.
        # Portfolio.update_fill logic: "fill_qty = event.quantity if BUY else -event.quantity"
        # So FillEvent quantity should be POSITIVE.
        
        fill_short = FillEvent(
            timestamp=None, symbol='NQ', 
            quantity=1, side=OrderSide.SELL, 
            price=100.0, commission=0.0, slippage=0.0
        )
        p.update_fill(fill_short)
        
        # Check Position
        pos = p.current_positions['NQ']
        self.assertEqual(pos.quantity, -1)
        self.assertEqual(pos.avg_price, 100.0)
        self.assertEqual(p.current_cash, 100000 - (-1*100)) # Cash increases by short sale proceeds? 
        # Standard accounting: Cash increases, but Liability increases.
        # Portfolio.py: self.current_cash -= (cost + comm). 
        # cost = -1 * 100 = -100. Cash -= -100 => Cash += 100. Correct.
        self.assertEqual(p.current_cash, 100100.0)
        
        # 2. Cover Short (Buy 1 NQ @ 90)
        fill_cover = FillEvent(
            timestamp=None, symbol='NQ',
            quantity=1, side=OrderSide.BUY,
            price=90.0, commission=0.0, slippage=0.0
        )
        p.update_fill(fill_cover)
        
        # Check Position
        pos = p.current_positions['NQ']
        self.assertEqual(pos.quantity, 0)
        
        # Check Log
        trade = p.trade_log[-1]
        print(f"Trade Log: {trade}")
        
        # Expected PnL: (100 - 90) * 1 = +10.
        self.assertEqual(trade['realized_pnl'], 10.0)
        
        # Final Cash
        # Initial: 100,000
        # Short: +100 (Cash 100,100)
        # Cover: -90 (Cash 100,010)
        # Expected: 100,010.
        self.assertEqual(p.current_cash, 100010.0)

    def test_short_loss(self):
        """Short 1 @ 100. Cover @ 110. Loss -10."""
        events = deque()
        class MockData:
             symbol_list = ['NQ']
             def get_latest_bar(self, s): return None
        p = Portfolio(bars=MockData(), events=events, initial_capital=100000)
        p.instrument_config = {'NQ': {'multiplier': 1.0}}
        
        # Short @ 100
        p.update_fill(FillEvent(None, 'NQ', 1, 100.0, 0.0, 0.0, OrderSide.SELL))
        
        # Cover @ 110 (Buy)
        p.update_fill(FillEvent(None, 'NQ', 1, 110.0, 0.0, 0.0, OrderSide.BUY))
        
        trade = p.trade_log[-1]
        print(f"Loss Trade: {trade}")
        self.assertEqual(trade['realized_pnl'], -10.0)

if __name__ == '__main__':
    unittest.main()
