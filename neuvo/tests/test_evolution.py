import pytest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn
from neuvo.evolution import EvolutionStrategy
from neuvo.model import NeuroevolutionTransformer
import random

class TestEvolutionStrategy(EvolutionStrategy):
    """Concrete implementation of EvolutionStrategy for testing"""
    def select_parents(self, population, tournament_size):
        if not population:
            return []
        # Return enough parents for the entire population
        num_parents_needed = len(population)
        parents = []
        while len(parents) < num_parents_needed:
            tournament = random.sample(population, min(tournament_size, len(population)))
            parents.append(tournament[0])
        return parents
    
    def crossover(self, parent1, parent2):
        # Get genotypes from parents and create child genotypes
        parent1_genotype = parent1.get_genotype()
        parent2_genotype = parent2.get_genotype()
        return parent1_genotype, parent2_genotype
    
    def mutate(self, genotype):
        return genotype

@pytest.fixture
def config():
    return {
        'mutation_rate': 0.1,
        'population_size': 4,
        'tournament_size': 2,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 2
    }

@pytest.fixture
def mock_transformer():
    transformer = Mock()
    
    # Create a sample genotype
    sample_genotype = {
        'state_dict': {
            'layer1.weight': torch.randn(10, 10),
            'layer1.bias': torch.randn(10)
        }
    }
    
    # Mock get_genotype method
    transformer.get_genotype.return_value = sample_genotype
    transformer.fitness = random.random()
    return transformer

@pytest.fixture
def evolution_strategy(config):
    return TestEvolutionStrategy(config)

@pytest.fixture
def population(mock_transformer):
    return [Mock(
        get_genotype=mock_transformer.get_genotype,
        fitness=random.random()
    ) for _ in range(4)]

def test_evolution_strategy_initialization(config):
    strategy = TestEvolutionStrategy(config)
    assert strategy.config == config
    assert strategy.config['mutation_rate'] == 0.1

def test_select_parents(evolution_strategy, population):
    parents = evolution_strategy.select_parents(population, tournament_size=2)
    assert len(parents) >= 2
    assert all(hasattr(p, 'get_genotype') for p in parents)

def test_crossover(evolution_strategy, mock_transformer):
    child1_genes, child2_genes = evolution_strategy.crossover(mock_transformer, mock_transformer)
    assert isinstance(child1_genes, dict)
    assert isinstance(child2_genes, dict)
    assert 'state_dict' in child1_genes
    assert 'state_dict' in child2_genes

def test_mutate(evolution_strategy):
    sample_genotype = {'state_dict': {'layer1.weight': torch.randn(10, 10)}}
    mutated = evolution_strategy.mutate(sample_genotype)
    assert isinstance(mutated, dict)
    assert 'state_dict' in mutated

def test_evolve_population_basic(evolution_strategy, population):
    new_pop = evolution_strategy.evolve_population(population)
    assert len(new_pop) == len(population)
    assert all(isinstance(p, dict) and 'state_dict' in p for p in new_pop)

def test_evolve_population_with_fittest(evolution_strategy, population, mock_transformer):
    new_pop = evolution_strategy.evolve_population(population, fittest=mock_transformer)
    assert len(new_pop) == len(population)
    # Check that first member is actually the fittest's genotype
    assert isinstance(new_pop[0], dict)
    assert 'state_dict' in new_pop[0]
    # Verify it matches the fittest's genotype
    assert new_pop[0] == mock_transformer.get_genotype()

def test_evolve_population_mutation_rates(config, population, mutation_rate=0.5):
    config['mutation_rate'] = mutation_rate
    strategy = TestEvolutionStrategy(config)
    
    with patch.object(strategy, 'mutate') as mock_mutate:
        mock_mutate.side_effect = lambda x: x
        new_pop = strategy.evolve_population(population)
        assert len(new_pop) == len(population)

def test_evolve_population_consistent_size(evolution_strategy, mock_transformer):
    for size in [2, 4, 6]:
        test_pop = [mock_transformer for _ in range(size)]
        new_pop = evolution_strategy.evolve_population(test_pop)
        assert len(new_pop) == size

def test_evolve_population_genotype_structure(evolution_strategy, population):
    new_pop = evolution_strategy.evolve_population(population)
    for individual in new_pop:
        assert isinstance(individual, dict)
        assert 'state_dict' in individual
        assert isinstance(individual['state_dict'], dict)
        assert all(isinstance(v, torch.Tensor) for v in individual['state_dict'].values())