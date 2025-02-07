"""Unit tests for the GA (Genetic Algorithm) class."""
import pytest
import numpy as np
from typing import Dict
from neuvo.evolution import GA  # Using the package import structure

@pytest.fixture
def base_ga():
    """Fixture for basic GA instance"""
    return GA(shape=(10, 10))

@pytest.fixture
def eco_ga():
    """Fixture for GA instance with eco mode enabled"""
    return GA(shape=(10, 10), eco=True)

def test_initialization(base_ga, eco_ga):
    """Test initialization of GA class"""
    # Test basic initialization
    assert base_ga.shape == (10, 10)
    assert base_ga.mutation_rate == 0.1
    assert base_ga.genes == 5
    assert not base_ga.eco
    
    # Test eco mode initialization
    assert eco_ga.genes == 9
    assert eco_ga.eco

def test_genotype_builder(base_ga):
    """Test genotype builder creates valid hyperparameters"""
    genotype = base_ga.genotype_builder()
    
    # Check if all required keys exist
    required_keys = {'learning_rate', 'optimizer', 'num_epochs', 
                    'batch_size', 'warmup_ratio', 'weight_decay'}
    assert all(key in genotype for key in required_keys)
    
    # Check if values are within expected ranges
    assert genotype['learning_rate'] in base_ga.learning_rates
    assert genotype['optimizer'] in base_ga.optimizers
    assert genotype['batch_size'] in base_ga.batch_sizes
    assert genotype['warmup_ratio'] in base_ga.warmup_ratios
    assert genotype['weight_decay'] in base_ga.weight_decays
    assert genotype['num_epochs'] >= 1

def test_eco_genotype_builder(eco_ga):
    """Test genotype builder with eco mode enabled"""
    genotype = eco_ga.genotype_builder()
    
    # Check additional eco mode parameters
    eco_keys = {'mutation_rate', 'population_size', 'cloning_rate', 'max_generations'}
    assert all(key in genotype for key in eco_keys)
    
    # Test value ranges for eco parameters
    assert 0 <= genotype['mutation_rate'] <= 1
    assert 3 <= genotype['population_size'] <= 13
    assert 0 <= genotype['cloning_rate'] <= 1
    assert 1 <= genotype['max_generations'] <= 201

def test_remove_metrics(base_ga):
    """Test removal of evaluation metrics"""
    # Add some metrics to phenotype
    base_ga.phenotype.update({
        'loss': 0.5,
        'accuracy': 0.8,
        'f1': 0.75,
        'precision': 0.7,
        'recall': 0.8,
        'eval_loss': 0.6,
        'eval_accuracy': 0.75,
        'wer': 0.2,
        'bleu': 0.6
    })
    
    base_ga.remove_metrics()
    metrics = {'loss', 'accuracy', 'f1', 'precision', 'recall',
              'eval_loss', 'eval_accuracy', 'wer', 'bleu'}
              
    # Check that no metrics remain in phenotype
    assert all(metric not in base_ga.phenotype for metric in metrics)

def test_mutate(base_ga):
    """Test mutation of hyperparameters"""
    # Store original values
    original_genotype = base_ga.genotype.copy()
    
    # Perform multiple mutations to increase chance of seeing a change
    for _ in range(10):
        base_ga.mutate()
        # Check if any value has changed from the original
        if any(original_genotype[k] != base_ga.genotype[k] 
              for k in original_genotype.keys()):
            break
    
    # Verify mutation maintains valid values
    assert base_ga.phenotype['learning_rate'] in base_ga.learning_rates
    assert base_ga.phenotype['optimizer'] in base_ga.optimizers
    assert base_ga.phenotype['batch_size'] in base_ga.batch_sizes
    assert base_ga.phenotype['warmup_ratio'] in base_ga.warmup_ratios
    assert base_ga.phenotype['weight_decay'] in base_ga.weight_decays
    assert base_ga.phenotype['num_epochs'] >= 1

def test_mutate_eco(eco_ga):
    """Test mutation with eco mode enabled"""
    original_genotype = eco_ga.genotype.copy()
    eco_ga.mutation_rate = 1.0  # Force mutation to occur
    
    eco_ga.mutate()
    
    # Check that eco-specific parameters maintain valid ranges after mutation
    assert 0 <= eco_ga.phenotype['mutation_rate'] <= 1
    assert eco_ga.phenotype['population_size'] >= 3
    assert 0 <= eco_ga.phenotype['cloning_rate'] <= 1
    assert eco_ga.phenotype['max_generations'] >= 1

def test_custom_phenotype():
    """Test initialization with custom phenotype"""
    custom_phenotype = {
        'learning_rate': 2e-5,
        'optimizer': 'Adam',
        'num_epochs': 3,
        'batch_size': 16,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01
    }
    
    ga = GA(shape=(10, 10), phenotype=custom_phenotype)
    assert ga.phenotype == custom_phenotype

def test_shape_handling():
    """Test initialization with different shape parameters"""
    # Test valid shape
    ga = GA(shape=(10, 10))
    assert ga.shape == (10, 10)
    
    # Test that GA handles or converts different shape formats
    ga2 = GA(shape=[10, 10])
    assert ga2.shape is not None
    
    # Test that GA can handle single dimension
    ga3 = GA(shape=10)
    assert ga3.shape is not None

def test_mutation_bounds():
    """Test that mutation maintains valid bounds for all parameters"""
    ga = GA(shape=(10, 10))
    
    # Run multiple mutations to increase chance of catching any bound violations
    for _ in range(10):
        ga.mutate()
        assert ga.phenotype['learning_rate'] in ga.learning_rates
        assert ga.phenotype['optimizer'] in ga.optimizers
        assert ga.phenotype['batch_size'] in ga.batch_sizes
        assert ga.phenotype['warmup_ratio'] in ga.warmup_ratios
        assert ga.phenotype['weight_decay'] in ga.weight_decays
        assert ga.phenotype['num_epochs'] >= 1

def test_genotype_consistency():
    """Test that genotype and phenotype remain consistent after operations"""
    ga = GA(shape=(10, 10))
    
    # After initialization
    assert ga.genotype == ga.phenotype
    
    # After mutation
    ga.mutate()
    assert ga.genotype == ga.phenotype
    
    # After removing metrics
    ga.phenotype.update({'loss': 0.5, 'accuracy': 0.8})
    ga.remove_metrics()
    assert ga.genotype == ga.phenotype