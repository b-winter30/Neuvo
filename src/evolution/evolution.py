from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from .ga import GA 
from ge import GE

class EvolutionStrategy:
    """Base class for evolution strategies"""
    def __init__(self, strategy_type: str, config: Dict[str, Any]):
        """
        Initialize evolution strategy
        
        Args:
            strategy_type: Either 'ga' or 'ge'
            config: Configuration dictionary with evolution parameters
        """
        self.strategy_type = strategy_type.lower()
        self.config = config
        self.strategy = self._create_strategy()

    def _create_strategy(self):
        """Create the appropriate evolution strategy"""
        if self.strategy_type == 'ga':
            return GA(
                shape=self.config.get('shape'),
                mutation_rate=self.config.get('mutation_rate', 0.1),
                eco=self.config.get('eco', False)
            )
        elif self.strategy_type == 'ge':
            return GE(
                shape=self.config.get('shape'),
                mutation_rate=self.config.get('mutation_rate', 0.1),
                user_grammar_file=self.config.get('grammar_file'),
                genotype_length=self.config.get('genotype_length', 32),
                gene_value=self.config.get('gene_value', 40)
            )
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")

    def mutate(self, individual: Any):
        """Perform mutation using the selected strategy"""
        return self.strategy.mutate(individual)

    def crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """Perform crossover using the selected strategy"""
        if hasattr(self.strategy, 'crossover'):
            return self.strategy.crossover(parent1, parent2)
        else:
            # Default crossover if not implemented in strategy
            child1 = parent1.copy()
            child2 = parent2.copy()
            return child1, child2

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters from the strategy"""
        return self.strategy.phenotype if hasattr(self.strategy, 'phenotype') else {}

    def remove_metrics(self):
        """Remove evaluation metrics from the strategy"""
        if hasattr(self.strategy, 'remove_metrics'):
            self.strategy.remove_metrics()