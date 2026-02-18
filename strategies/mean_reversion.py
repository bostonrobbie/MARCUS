from backtesting.strategy import Strategy
try:
    import pandas_ta as ta
except ImportError:
    ta = None

class MeanReversion(Strategy):
    """
    Mean Reversion Strategy using Bollinger Bands.
    
    Logic:
    - Buy when Price < Lower Band (Oversold).
    - Sell/Exit when Price > MA (Mean).
    """
    
    def __init__(self, data_handler, events, lookback=20, std_dev=2.0):
        super().__init__(data_handler, events)
        self.lookback = lookback
        self.std_dev = std_dev

    def calculate_signals(self, event):
        symbol = event.symbol
        
        # Calculate Indicators (Lazy calculation or Pre-calc?)
        # SmartDataHandler + pandas-ta allows access.
        # But we need 'history' to calculate TA.
        # self.bars.get_latest_n(symbol, N) returns list of Bars. 
        # Easier: accessing the full DF via symbol_data if using SmartDataHandler.
        # Note: In a real Event loop, re-calculating on full DF every step is slow.
        # Optimized way: Rolling calculation.
        # For this demo: We access the dataframe and get the latest row's value.
        
        # Assumption: SmartDataHandler updates the underlying df or we just calculate on what we have?
        # Actually, Strategy.bars is the DataHandler.
        # Let's assume Pre-Calculation for speed or just accessing `.ta` if supported.
        
        # Accessing underlying dataframe from DataHandler (Backdoor for TA)
        if not hasattr(self.bars, 'symbol_data'):
            return 
            
        df = self.bars.symbol_data[symbol]
        # We need the index of current bar. 
        # 'event.timestamp' is the key.
        
        try:
            # Check if indicators exist, else calc
            if 'BBL_20_2.0' not in df.columns:
                df.ta.bbands(length=self.lookback, std=self.std_dev, append=True)
            
            # Get current values
            current_bar = df.loc[event.timestamp]
            lower_band = current_bar[f'BBL_{self.lookback}_{self.std_dev}']
            mid_band = current_bar[f'BBM_{self.lookback}_{self.std_dev}']
            
            price = event.close
            
            # Entry Logic
            if not self.bought and price < lower_band:
                self.buy(symbol, 100) # Buy 100 shares
                
            # Exit Logic
            elif self.bought and price > mid_band:
                self.exit(symbol)
                
        except KeyError:
            # Data might not be aligned or timestamp missing
            pass
        except Exception as e:
            # Indicator calculation error
            pass
