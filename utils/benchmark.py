import time
from runner import run_strategy, run_vector_strategy
from examples.trend_following import MovingAverageCrossover

def run_benchmark():
    symbol_list = ['NVDA']
    data_dirs = ['./examples']
    params = {'short_window': 20, 'long_window': 50}
    interval = '1m'
    
    print("\n" + "="*50)
    print("BACKTEST ENGINE BENCHMARK (1m Data)")
    print("="*50)
    
    # 1. Event-Driven Engine
    start_event = time.time()
    run_strategy(MovingAverageCrossover, symbol_list, data_dirs, plot=False, verbose=False, interval=interval, **params)
    end_event = time.time()
    event_time = end_event - start_event
    print(f"[*] Event-Driven Engine Time: {event_time:.4f}s")
    
    # 2. Vectorized Engine
    start_vector = time.time()
    run_vector_strategy(MovingAverageCrossover, symbol_list, data_dirs, interval=interval, **params)
    end_vector = time.time()
    vector_time = end_vector - start_vector
    print(f"[*] Vectorized Engine Time:   {vector_time:.4f}s")
    
    speedup = event_time / vector_time if vector_time > 0 else 0
    print("-" * 50)
    print(f"RESULT: Vectorized Engine is {speedup:.1f}x faster.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_benchmark()
