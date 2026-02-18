import os
import time
import pandas as pd
import numpy as np
try:
    import yfinance as yf
except ImportError:
    yf = None

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Generator
from datetime import datetime
from .schema import Bar
from .monitor import PipelineMonitor

class DataHandler(ABC):
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> List[Bar]:
        raise NotImplementedError

    @abstractmethod
    def update_bars(self) -> bool:
        raise NotImplementedError

class SmartDataHandler(DataHandler):
    """
    Smart Data Handler:
    1. Looks for CSV in local directories.
    2. If not found, downloads from Yahoo Finance.
    3. Caches downloads? (Maybe V2).
    """

    def __init__(self, symbol_list: List[str], search_dirs: List[str] = None, 
                 start_date: datetime = None, end_date: datetime = None, 
                 interval: str = '1d'):
        self.symbol_list = symbol_list
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.interval = interval
        
        # Default search paths
        self.search_dirs = search_dirs if search_dirs else []
        
        # Add User's Specific Custom Path (Now handled by config passed in)
        # self.search_dirs.append(r"C:\Users\User\Desktop\Portfolio\OHLC\Intra OHLC")
        
        # Add current directory and examples
        self.search_dirs.append(os.path.join(os.getcwd(), 'examples'))
        self.search_dirs.append(os.getcwd())
        
        # Internal data structures
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.latest_symbol_data: Dict[str, List[Bar]] = {}
        self.continue_backtest = True
        self._bar_generators: Dict[str, Generator] = {}
        
        self._load_data()

    def _load_data(self):
        for symbol in self.symbol_list:
            df = self._fetch_data(symbol)
            
            if df is None or df.empty:
                raise ValueError(f"Could not find or download data for {symbol}")

            # Standardize Columns
            df.columns = [c.capitalize() for c in df.columns]
            
            # Ensure Date Index
            # CRITICAL: Preserve ET wall clock times (9:30 AM ET stays 9:30)
            # which the ORB strategy and all session-time logic depends on.
            # Our CSV timestamps like "2010-06-02 18:05:00 -04:00" already
            # represent ET local time with a UTC offset suffix. We strip the
            # offset and parse the first 19 chars directly (fast path).
            # Fallback to the slower utc→tz_convert path for other formats.
            date_col = next((c for c in df.columns if c in ['Date', 'Datetime', 'Time']), None)
            if date_col:
                sample = str(df[date_col].iloc[0])
                # Fast path: timestamps with UTC offset suffix like "2010-06-02 18:05:00 -04:00"
                # The local time portion (first 19 chars) is already ET wall clock time
                if len(sample) > 19 and ('+' in sample[19:] or '-' in sample[19:]):
                    df[date_col] = pd.to_datetime(df[date_col].astype(str).str[:19], format='%Y-%m-%d %H:%M:%S')
                else:
                    # Fallback: parse with timezone, convert to ET
                    raw_dt = pd.to_datetime(df[date_col], utc=True)
                    df[date_col] = raw_dt.dt.tz_convert('America/New_York').dt.tz_localize(None)
                df.set_index(date_col, inplace=True)
            
            # Ensure Volume column exists (some CSVs lack it)
            if 'Volume' not in df.columns:
                df['Volume'] = 0.0

            # Ensure sorting
            df.sort_index(inplace=True)
            
            # Remove timezone if exists (safety net)
            # Always convert to ET first to preserve wall clock times
            if hasattr(df.index, 'tz_localize'):
                if df.index.tz is not None:
                    df.index = df.index.tz_convert('America/New_York').tz_localize(None)
                # else: already naive (ET wall clock), leave as-is

            # --- Resampling Logic ---
            # If current data frequency doesn't match requested interval
            # Note: We assume input data freq can be inferred or specified
            if self.interval and self.interval != '1d':
                # Map interval like '15m' to pandas freq '15min'
                # Use regex to only replace trailing 'm' (not 'min', 'max', etc.)
                import re
                freq = re.sub(r'(\d+)m$', r'\1min', self.interval).replace('h', 'H')
                
                # Check current frequency (crude check)
                if len(df) > 1:
                    actual_diff = (df.index[1] - df.index[0]).total_seconds() / 60
                    target_diff = pd.to_timedelta(freq).total_seconds() / 60
                    
                    if actual_diff < target_diff:
                        print(f"Resampling data for {symbol} from {actual_diff}m to {self.interval}...")
                        logic = {
                            'Open' : 'first',
                            'High' : 'max',
                            'Low'  : 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }
                        # Filter existing columns to avoid errors
                        existing_logic = {k: v for k, v in logic.items() if k in df.columns}
                        df = df.resample(freq).apply(existing_logic).dropna()

            # Filtering Date Range
            if self.start_date:
                df = df[df.index >= self.start_date]
            if self.end_date:
                df = df[df.index <= self.end_date]

            if df.empty:
                 print(f"Warning: Data for {symbol} is empty after filtering {self.start_date} -> {self.end_date}")

            self.symbol_data[symbol] = df
            self.latest_symbol_data[symbol] = []
            self._bar_generators[symbol] = df.iterrows()

    def _fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        # 1. Search Local Files (Unless Forced Download)
        # TODO: Add 'force_download' param to init if needed
        
        monitor = PipelineMonitor()
        
        # Priority Search
        paths_to_check = []
        for d in self.search_dirs:
            if not os.path.exists(d): continue
            
            # Interval specifics
            if self.interval:
                int_map = {'1m': 'm1', '5m': 'm5', '15m': 'm15'}
                suffix = int_map.get(self.interval)
                if suffix:
                    paths_to_check.append(os.path.join(d, f"A2API-{symbol.upper()}-{suffix}.csv"))
            
            # Generics
            paths_to_check.extend([
                os.path.join(d, f"{symbol}.csv"),
                os.path.join(d, f"A2API-{symbol.upper()}-m1.csv"),
                os.path.join(d, f"{symbol.lower()}.csv"),
                os.path.join(d, f"{symbol.upper()}.csv")
            ])
            
        for p in paths_to_check:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    monitor.log_data_loading(symbol, "LOCAL_FILE", True)
                    return df
                except Exception as e:
                    monitor.log_event("DataHandler", "READ_ERROR", f"Failed to read {p}: {e}", "ERROR")

        # 2. Cache management for yfinance
        monitor.log_event("DataHandler", "CACHE_CHECK", f"Checking cache for {symbol}...")
        cache_dir = os.path.join(os.getcwd(), 'cache', self.interval)
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{symbol}.csv")
        
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            age_sec = time.time() - mtime
            max_age = 3600 if self.interval in ['1m', '5m', '15m'] else 86400
            
            if age_sec < max_age:
                try:
                    df = pd.read_csv(cache_path)
                    monitor.log_data_loading(symbol, "CACHE", True)
                    return df
                except Exception:
                    pass

        # 3. Download from yfinance
        monitor.log_event("DataHandler", "DOWNLOAD_START", f"Downloading {symbol} ({self.interval}) from Yahoo Finance...")
        
        if yf is None:
            monitor.log_event("DataHandler", "MODULE_MISSING", "yfinance not installed", "CRITICAL")
            return None

        try:
            period = "max"
            if self.interval == '1m': period = "7d"
            elif self.interval in ['5m', '15m']: period = "60d"
                
            data = yf.download(symbol, period=period, interval=self.interval, progress=False)
            if not data.empty:
                # Handle MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    if symbol in data.columns.levels[1]:
                         data = data.xs(symbol, level=1, axis=1)
                
                # Cache cleaned data
                data.to_csv(cache_path)
                data.reset_index(inplace=True)
                
                monitor.log_data_loading(symbol, "YFINANCE", True)
                return data
            else:
                monitor.log_data_loading(symbol, "YFINANCE", False)
                
        except Exception as e:
            monitor.log_event("DataHandler", "DOWNLOAD_ERROR", f"Failed to download {symbol}: {e}", "ERROR")
        
        return None

    def _get_new_bar(self, symbol: str) -> Optional[Bar]:
        try:
            index, row = next(self._bar_generators[symbol])
            return Bar(
                symbol=symbol,
                timestamp=index,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row.get('Volume', 0.0) if hasattr(row, 'get') else (row['Volume'] if 'Volume' in row.index else 0.0)
            )
        except StopIteration:
            return None

    def update_bars(self) -> bool:
        any_updates = False
        for symbol in self.symbol_list:
            bar = self._get_new_bar(symbol)
            if bar is not None:
                self.latest_symbol_data[symbol].append(bar)
                any_updates = True
        
        if not any_updates:
            self.continue_backtest = False
            
        return any_updates

    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        bars_list = self.latest_symbol_data.get(symbol)
        return bars_list[-1] if bars_list else None

    def get_latest_bars(self, symbol: str, N: int = 1) -> List[Bar]:
        bars_list = self.latest_symbol_data.get(symbol)
        return bars_list[-N:] if bars_list else []


class MemoryDataHandler(DataHandler):
    """
    In-Memory Data Handler for Optimization.
    Receives pre-loaded DataFrames directly.
    """
    def __init__(self, symbol_data: Dict[str, pd.DataFrame]):
        self.symbol_data = symbol_data
        self.symbol_list = list(symbol_data.keys())  # FIX: was missing, breaks Strategy/Portfolio/Engine
        self.latest_symbol_data: Dict[str, List[Bar]] = {s: [] for s in symbol_data.keys()}
        self.continue_backtest = True
        self._bar_generators: Dict[str, Generator] = {s: df.iterrows() for s, df in symbol_data.items()}

    def _get_new_bar(self, symbol: str) -> Optional[Bar]:
        try:
            index, row = next(self._bar_generators[symbol])
            return Bar(
                symbol=symbol,
                timestamp=index,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row.get('Volume', 0.0) if hasattr(row, 'get') else (row['Volume'] if 'Volume' in row.index else 0.0)
            )
        except StopIteration:
            return None

    def update_bars(self) -> bool:
        any_updates = False
        for symbol in self.symbol_list:
            bar = self._get_new_bar(symbol)
            if bar is not None:
                self.latest_symbol_data[symbol].append(bar)
                any_updates = True
        
        if not any_updates:
            self.continue_backtest = False
            
        return any_updates

    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        bars_list = self.latest_symbol_data.get(symbol)
        return bars_list[-1] if bars_list else None

    def get_latest_bars(self, symbol: str, N: int = 1) -> List[Bar]:
        bars_list = self.latest_symbol_data.get(symbol)
        return bars_list[-N:] if bars_list else []
