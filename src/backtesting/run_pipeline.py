
from .pipeline import ResearchPipeline
from .strategies import NqOrb15m
from .vector_engine import VectorizedNQORB
from datetime import datetime

# Configuration
SYMBOL_LIST = ['NQ=F'] # Or a generic ticker like NQ
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2023, 1, 1)

# Parameter Grid for Optimization
PARAM_GRID = {
    'ema_filter': [20, 50],
    'sl_atr_mult': [1.5, 2.0],
    'tp_atr_mult': [3.0, 4.0]
}

def main():
    pipeline = ResearchPipeline(
        strategy_name="NqOrb15m",
        strategy_cls=NqOrb15m,
        vector_strategy_cls=VectorizedNQORB,
        symbol_list=SYMBOL_LIST,
        param_grid=PARAM_GRID,
        data_range_start=START_DATE,
        data_range_end=END_DATE,
        interval='15m',
        search_dirs=[r"C:\Users\User\Documents\AI\backtesting\data"] # Adjust if needed
    )
    
    pipeline.execute()

if __name__ == "__main__":
    main()
