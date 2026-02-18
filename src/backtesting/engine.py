import queue
import time
from typing import Optional
from .schema import Bar, SignalEvent, OrderEvent, FillEvent
from .data import DataHandler
from .strategy import Strategy
from .portfolio import Portfolio
from .execution import ExecutionHandler

class BacktestEngine:
    """
    Encapsulates the settings and components for carrying out an event-driven backtest.
    """
    
    def __init__(self, 
                 data_handler: DataHandler, 
                 strategy: Strategy, 
                 portfolio: Portfolio, 
                 execution_handler: ExecutionHandler):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.events = portfolio.events # Use the same queue shared by components

    def run(self):
        """
        Executes the backtest simulation.
        """
        print("Starting Backtest...")
        
        while True:
            # 1. Update Data (Outer Loop = Time Step)
            if self.data_handler.update_bars():
                # Push Market Events (Bars) to Queue
                for symbol in self.data_handler.symbol_list:
                    bar = self.data_handler.get_latest_bar(symbol)
                    if bar:
                        self.events.put(bar)
            else:
                # No more data
                break

            # 2. Process all events for this Time Step
            while not self.events.empty():
                event = self.events.get()
                
                if isinstance(event, Bar):
                    # 1. First, check match for pending orders (Market On Open)
                    # Use getattr to be safe if using different ExecutionHandlers, or enforce strict interface
                    if hasattr(self.execution_handler, 'on_bar'):
                        self.execution_handler.on_bar(event)

                    # 2. Then Notify Strategy (Calculate new signals)
                    self.strategy.calculate_signals(event)
                    
                elif isinstance(event, SignalEvent):
                    self.portfolio.update_signal(event)
                    
                elif isinstance(event, OrderEvent):
                    self.execution_handler.execute_order(event)
                    
                elif isinstance(event, FillEvent):
                    self.portfolio.update_fill(event)
                    self.strategy.on_fill(event)

            # 3. End of Time Step: Update Portfolio Statistics (Mark-to-Market)
            self.portfolio.update_timeindex()
            
        print("Backtest Completed.")
