from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from ..model import NeuroevolutionTransformer
import random

class EvolutionStrategy(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def select_parents(self, population: List[NeuroevolutionTransformer], tournament_size: int) -> List[NeuroevolutionTransformer]:
        """Select parents from population"""
        pass

    @abstractmethod
    def crossover(self, parent1: NeuroevolutionTransformer, parent2: NeuroevolutionTransformer) -> Tuple[Dict, Dict]:
        """Perform crossover between two parents"""
        pass

    @abstractmethod
    def mutate(self, genotype: Dict) -> Dict:
        """Mutate a single genotype"""
        pass

    def evolve_population(
        self, 
        population: List[NeuroevolutionTransformer],
        fittest: NeuroevolutionTransformer = None,
        tournament_size: int = 3
    ) -> List[Dict]:  # Changed return type to List[Dict]
        """Evolve the entire population
        
        Args:
            population: List of NeuroevolutionTransformer instances
            fittest: Optional fittest individual to preserve
            tournament_size: Size of tournament for parent selection
            
        Returns:
            List of genotype dictionaries for the new population
        """
        new_population = []
        
        # Elitism - convert fittest to genotype
        if fittest:
            new_population.append(fittest.get_genotype())

        # Select parents
        parents = self.select_parents(population, tournament_size)
        
        # Create offspring
        while len(new_population) < len(population):
            if len(parents) < 2:  # If we run out of parents, select more
                parents.extend(self.select_parents(population, tournament_size))
            
            parent1, parent2 = parents[0:2]
            parents = parents[2:]  # Remove used parents
            
            # Crossover
            child1_genes, child2_genes = self.crossover(parent1, parent2)
            
            # Mutation
            if random.random() < self.config['mutation_rate']:
                child1_genes = self.mutate(child1_genes)
            if random.random() < self.config['mutation_rate']:
                child2_genes = self.mutate(child2_genes)
            
            # Add to new population
            new_population.extend([child1_genes, child2_genes])
        
        return new_population[:len(population)]  # Ensure consistent population size