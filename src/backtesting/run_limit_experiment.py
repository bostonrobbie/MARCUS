
from .pipeline import ResearchPipeline
from .strategies_limit import NqPullbackLimit
# import vector strategy or use None if strictly testing Event Logic basics first
from .vector_engine import VectorizedNQORB 
from datetime import datetime

# Configuration
SYMBOL_LIST = ['NQ=F'] 
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2023, 1, 1)

# Parameter Grid
PARAM_GRID = {
    'ema_filter': [50], # Just test one config for now to verify Limit Logic
    'orb_start': ["09:30"],
    'orb_end': ["09:45"]
}

def main():
    print("Running LIMIT ORDER Strategy Pipeline...")
    # Note: We are using ResearchPipeline but our NqPullbackLimit doesn't have a Vector equivalent yet.
    # The pipeline executes Vector optimization first. 
    # For this specific "Plan/Dev" phase, let's bypass the vector optimization requirement 
    # OR we can just run the Event Engine check manually if the Pipeline insists on Vector parity.
    
    # Ideally, we should create a VectorizedPullbackLimit to optimize parameters FAST.
    # But for now, let's just see if the concept logic runs in the Event Engine.
    
    # Using the existing pipeline will attempt to run VectorizedNQORB (Breakout) then NqPullbackLimit (Pullback).
    # This mismatch is fine if we just want to test the machinery.
    
    pipeline = ResearchPipeline(
        strategy_name="NqPullbackLimit",
        strategy_cls=NqPullbackLimit,
        vector_strategy_cls=VectorizedNQORB, # Proxy
        symbol_list=SYMBOL_LIST,
        param_grid=PARAM_GRID,
        data_range_start=START_DATE,
        data_range_end=END_DATE,
        interval='15m',
        search_dirs=[r"C:\Users\User\Documents\AI\backtesting\data"]
    )
    
    pipeline.execute()

if __name__ == "__main__":
    main()
