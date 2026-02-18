
import pandas as pd
import random
from typing import List, Type, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .factory import StrategyFactory, StrategyGenome
from .data import DataHandler, MemoryDataHandler
from .vector_engine import VectorEngine, VectorizedNQORB # We'll need a Generic Vector Strategy later

# For now, we assume VectorizedNQORB can accept ANY params from the Genome.
# Realistically, we need a 'UniversalVectorStrategy' that switches logic based on params.
# I will alias it to NqOrb for the prototype, utilizing its flags.

class EvolutionaryOptimizer:
    """
    Genetic Algorithm for Strategy Discovery.
    """
    def __init__(self,
                 data_handler_cls: Type[DataHandler],
                 data_handler_args: tuple,
                 population_size: int = 50,
                 generations: int = 10,
                 mutation_rate: float = 0.2,
                 initial_capital: float = 100000.0,
                 n_jobs: int = -1):
        
        self.data_handler_cls = data_handler_cls
        self.data_handler_args = data_handler_args
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.initial_capital = initial_capital
        
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        
        self.factory = StrategyFactory()
        self.population: List[StrategyGenome] = []
        self.history = []

    def initialize_population(self):
        print(f"  🧬 Spawning Initial Population ({self.pop_size})...")
        self.population = [self.factory.generate_random_genome() for _ in range(self.pop_size)]

    def evaluate_fitness(self) -> List[Dict]:
        """
        Run Backtests for the entire population.
        Returns list of results dicts.
        """
        # 1. Load Data Once
        loader = self.data_handler_cls(*self.data_handler_args)
        symbol = self.data_handler_args[0][0]
        if not loader.symbol_data: return []
        df = loader.symbol_data[symbol]
        
        results = []
        
        # 2. Parallel Execution
        # We use a helper function similar to GridSearch
        from .optimizer import _run_single_vector_backtest
        # We need to map factory genes -> Numba friendly params
        # Note: VectorizedNQORB expects specific args. 
        # Our Factory generates a dict. We pass that dict as **kwargs.
        # VectorizedNQORB.__init__ handles kwargs but only sets what it knows.
        # We need to ensure Genome keys match VectorStrategy args.
        
        args_list = []
        for genome in self.population:
            params = genome.to_params()
            # Map Genome Keys to VectorizedNQORB Keys if needed
            # For now, Factory keys (sl_mult, etc.) match closely.
            # Entry logic (ORB vs RSI) needs to be handled by the Strategy Class.
            # If logic is 'RSI', we need to tell VectorStrategy to use RSI.
            # Currently VectorizedNQORB is mostly ORB. 
            # We might need to refactor VectorizedNQORB to accept 'entry_mode' string?
            # Or map 'entry_logic'='RSI' -> use_rvol=False, use_rsi=True?
            
            args = (VectorEngine, VectorizedNQORB, params, self.initial_capital, df)
            args_list.append(args)

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(_run_single_vector_backtest, arg): arg[2] for arg in args_list}
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                
        return results

    def select_survivors(self, results: List[Dict]) -> List[StrategyGenome]:
        """Tournament Selection or Rank Based."""
        # Sort by Sharpe or Return
        # Simple: Total Return
        results.sort(key=lambda x: x.get('Total Return', -999), reverse=True)
        
        # Keep Top 20% Elites
        elite_count = int(self.pop_size * 0.2)
        top_results = results[:elite_count]
        
        # Recover Genomes from results (assuming params roughly identify them)
        # Better: Carry ID. 
        # For prototype, we reconstruct from params or just keep indices?
        # Actually evaluated results are decoupled from population list order in parallel.
        # We need to map back.
        # Let's assume we can map back via params match.
        
        survivors = []
        for res in top_results:
            # Reconstruct genome from successful params
            # Note: evaluate_fitness modified params? No.
            survivors.append(StrategyGenome(res)) # Create new Genome object from dict
            
        return survivors

    def evolve(self, survivors: List[StrategyGenome]):
        """Create next generation."""
        next_gen = []
        
        # 1. Elitism (Keep Survivors)
        next_gen.extend(survivors)
        
        # 2. Fill rest
        while len(next_gen) < self.pop_size:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = self.factory.crossover(p1, p2)
            child = self.factory.mutate(child, self.mutation_rate)
            next_gen.append(child)
            
        self.population = next_gen

    def run(self) -> pd.DataFrame:
        print(f"Starting EVOLUTIONARY OPTIMIZATION ({self.generations} Gens)...")
        self.initialize_population()
        
        best_overall = None
        
        for g in range(self.generations):
            print(f"  🔄 Generation {g+1}/{self.generations}...")
            
            # Evaluate
            results = self.evaluate_fitness()
            if not results: break
            
            # Stats
            avg_ret = sum(r.get('Total Return', 0) for r in results) / len(results)
            best_gen = max(results, key=lambda x: x.get('Total Return', -999))
            
            print(f"     Best: {best_gen.get('Total Return', 0):.2%} | Avg: {avg_ret:.2%}")
            
            # Log
            self.history.append({
                'generation': g,
                'avg_return': avg_ret,
                'best_return': best_gen.get('Total Return', 0),
                'best_params': best_gen
            })
            
            if best_overall is None or best_gen.get('Total Return') > best_overall.get('Total Return'):
                best_overall = best_gen
            
            # Select & Breed
            survivors = self.select_survivors(results)
            self.evolve(survivors)
            
        return pd.DataFrame([best_overall] if best_overall else [])

