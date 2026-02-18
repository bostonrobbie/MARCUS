
import random
from typing import Dict, List, Any, Optional

class StrategyGenome:
    """
    Represents the DNA of a trading strategy.
    Encodes:
    - Entry Trigger (e.g. RSI Cross)
    - Filter 1 (e.g. EMA Trend)
    - Filter 2 (e.g. ADX)
    - Exit Mode (e.g. Trailing Stop)
    - Parameters for all above
    """
    def __init__(self, genes: Dict[str, Any]):
        self.genes = genes
    
    def to_params(self) -> Dict[str, Any]:
        """Convert genes to flat param dict for VectorEngine."""
        return self.genes

    def __repr__(self):
        return f"Genome({self.genes})"

class StrategyFactory:
    """
    Generates random Strategy Genomes.
    Acts as the 'Creator' for the Genetic Algorithm.
    """
    
    def __init__(self):
        self.registry = {
            'entry_type': ['ORB', 'RSI_Cross', 'MA_Cross', 'Bollinger_Breakout'],
            'filter_type': ['None', 'EMA_Trend', 'RVOL', 'ADX_Regime'],
            'exit_type': ['Fixed_RR', 'Trailing_ATR', 'Time_Exit'],
            # Parameter Ranges
            'ema_period': range(10, 201, 10),
            'rsi_period': range(7, 22, 1),
            'atr_period': range(10, 31, 1),
            'sl_mult': [1.0, 1.5, 2.0, 2.5, 3.0],
            'tp_mult': [2.0, 3.0, 4.0, 5.0, 6.0],
        }

    def generate_random_genome(self) -> StrategyGenome:
        """Create a completely random strategy."""
        genes = {}
        
        # 1. Structure Selection
        genes['entry_logic'] = random.choice(self.registry['entry_type'])
        genes['filter_logic'] = random.choice(self.registry['filter_type'])
        genes['exit_logic'] = random.choice(self.registry['exit_type'])
        
        # 2. Param Selection
        genes['ema_period'] = random.choice(self.registry['ema_period'])
        genes['rsi_period'] = random.choice(self.registry['rsi_period'])
        genes['atr_period'] = random.choice(self.registry['atr_period'])
        genes['sl_mult'] = random.choice(self.registry['sl_mult'])
        genes['tp_mult'] = random.choice(self.registry['tp_mult'])
        
        # 3. Specific Flags based on logic
        genes['use_ema_filter'] = (genes['filter_logic'] == 'EMA_Trend')
        genes['use_rvol_filter'] = (genes['filter_logic'] == 'RVOL')
        genes['use_adx_filter'] = (genes['filter_logic'] == 'ADX_Regime')
        genes['use_trailing_stop'] = (genes['exit_logic'] == 'Trailing_ATR')
        
        return StrategyGenome(genes)

    def crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """Mix two genomes."""
        genes1 = parent1.genes
        genes2 = parent2.genes
        child_genes = {}
        
        # Randomly inherit each gene
        for key in genes1.keys():
            child_genes[key] = random.choice([genes1[key], genes2[key]])
            
        return StrategyGenome(child_genes)

    def mutate(self, genome: StrategyGenome, mutation_rate: float = 0.1) -> StrategyGenome:
        """Randomly modify genes."""
        genes = genome.genes.copy()
        
        for key in genes.keys():
            if random.random() < mutation_rate:
                # Modifiable Ranges
                if key == 'ema_period': genes[key] = random.choice(self.registry['ema_period'])
                elif key == 'sl_mult': genes[key] = random.choice(self.registry['sl_mult'])
                # ... extend for all keys
                elif key in self.registry: # If it's a categorical choice
                    genes[key] = random.choice(self.registry[key])

        return StrategyGenome(genes)
