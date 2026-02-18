import time
import pandas as pd
from backtesting.optimizer import VectorizedGridSearch
from backtesting.data import SmartDataHandler
from backtesting.vector_engine import VectorizedNQORB
from examples.nqorb_15m import NqOrb15m
from backtesting.portfolio import Portfolio
from backtesting.engine import BacktestEngine
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
import queue

def benchmark_optimization():
    print("Benchmarking Optimization (Vectorized)...")
    
    # Setup Data
    symbol_list = ['NQ']
    data_dir = r"C:\Users\User\Desktop\Portfolio\OHLC\Intra OHLC"
    # Use a shorter interval or subset if possible, but for benchmark we use actual data
    # We'll limit time range to 1 year to be fast
    start_date = pd.Timestamp("2020-01-01")
    end_date = pd.Timestamp("2021-01-01")
    
    # 64 Combinations
    param_grid = {
        'sl_atr_mult': [1.0, 2.0, 3.0, 4.0],
        'tp_atr_mult': [2.0, 4.0, 6.0, 8.0],
        'ema_filter': [50, 200],
        'atr_max_mult': [1.5, 2.5]
    }
    
    start_time = time.time()
    optimizer = VectorizedGridSearch(
        data_handler_cls=SmartDataHandler,
        data_handler_args=(symbol_list, [data_dir], start_date, end_date, '15m'),
        strategy_cls=NqOrb15m, # distinct from vector strategy but used for metadata
        param_grid=param_grid,
        initial_capital=100000.0
    )
    # Mock the strategy map if needed or ensure NqOrb15m maps to VectorizedNQORB
    # The current optimizer.py has a hardcoded map: 'NqOrb15m': VectorizedNQORB
    
    results = optimizer.run()
    duration = time.time() - start_time
    print(f"Optimization Verification: {len(results)} combinations")
    print(f"Optimization Time: {duration:.4f} seconds")
    return duration

def benchmark_certification_year():
    print("\nBenchmarking Certification (Event-Driven, 1 Year)...")
    start_time = time.time()
    
    symbol_list = ['NQ']
    data_dir = r"C:\Users\User\Desktop\Portfolio\OHLC\Intra OHLC"
    start_date = pd.Timestamp("2020-01-01")
    end_date = pd.Timestamp("2021-01-01")
    
    events = queue.Queue()
    data = SmartDataHandler(symbol_list, search_dirs=[data_dir], start_date=start_date, end_date=end_date, interval='15m')
    
    portfolio = Portfolio(data, events, initial_capital=100000.0)
    strategy = NqOrb15m(data, events, sl_atr_mult=2.0, tp_atr_mult=4.0, verbose=False)
    execution = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(1.0))
    engine = BacktestEngine(data, strategy, portfolio, execution)
    
    engine.run()
    
    duration = time.time() - start_time
    print(f"Certification Time: {duration:.4f} seconds")
    return duration

if __name__ == "__main__":
    t_opt = benchmark_optimization()
    t_cert = benchmark_certification_year()
    print(f"\nTotal Benchmark Time: {t_opt + t_cert:.4f} seconds")
