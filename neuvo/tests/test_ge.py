"""Unit tests for the Grammatical Evolution (GE) class."""
import pytest
from unittest.mock import MagicMock
from neuvo.evolution import GE

@pytest.fixture
def mocked_ge(monkeypatch):
    """Create a GE instance with mocked build method"""
    def mock_build(self):
        """Mock build that sets a valid phenotype"""
        self.phenotype = {
            'learning_rate': 2e-5,
            'optimizer': 'AdamW',
            'num_epochs': 3,
            'batch_size': 16,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01
        }
        return self

    def mock_phenotype_builder(self, *args, **kwargs):
        """Mock phenotype builder"""
        return self

    def mock_dictionise(self):
        """Mock dictionise"""
        return self

    # Apply mocks
    monkeypatch.setattr(GE, "build", mock_build)
    monkeypatch.setattr(GE, "phenotype_builder", mock_phenotype_builder)
    monkeypatch.setattr(GE, "dictionise", mock_dictionise)
    
    return GE(shape=(10, 10))

def test_initialization_parameters(mocked_ge):
    """Test basic initialization parameters"""
    assert mocked_ge.shape == (10, 10)
    assert mocked_ge.layers == 4
    assert mocked_ge.nodes == 8
    assert mocked_ge.mutation_rate == 0.1
    assert mocked_ge.genotype_length == 32
    assert mocked_ge.gene_value == 40
    assert mocked_ge.genes == 5

def test_custom_initialization(monkeypatch):
    """Test initialization with custom parameters"""
    def mock_build(self):
        self.phenotype = {
            'learning_rate': 2e-5,
            'optimizer': 'AdamW',
            'num_epochs': 3,
            'batch_size': 16,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01
        }
        return self

    monkeypatch.setattr(GE, "build", mock_build)
    monkeypatch.setattr(GE, "phenotype_builder", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(GE, "dictionise", lambda self: self)

    custom_ge = GE(
        shape=(10, 10),
        num_layers=6,
        num_nodes=12,
        mutation_rate=0.2,
        genotype_length=40,
        gene_value=50
    )
    
    assert custom_ge.layers == 6
    assert custom_ge.nodes == 12
    assert custom_ge.mutation_rate == 0.2
    assert custom_ge.genotype_length == 40
    assert custom_ge.gene_value == 50

def test_genotype_initial_state(mocked_ge):
    """Test initial genotype structure"""
    assert isinstance(mocked_ge.genotype, list)
    assert len(mocked_ge.genotype) == mocked_ge.genotype_length
    assert all(isinstance(x, int) for x in mocked_ge.genotype)
    assert all(0 <= x < mocked_ge.gene_value for x in mocked_ge.genotype)

def test_default_grammar(mocked_ge):
    """Test default grammar structure"""
    assert isinstance(mocked_ge.grammar, dict)
    
    # Test required grammar sections exist
    required_sections = {
        'start', 'expr', 'learning_rate', 'optimizer',
        'epochs', 'batch_size', 'warmup_ratio', 'weight_decay'
    }
    assert all(section in mocked_ge.grammar for section in required_sections)

def test_grammar_exclusions(mocked_ge):
    """Test grammar exclusions and operators"""
    mocked_ge.catch_grammar_exclusions()
    
    # Test operators
    assert hasattr(mocked_ge, 'basic_ops')
    assert set(mocked_ge.basic_ops) == {'+', '-', '*', '/'}
    
    # Test keys
    assert hasattr(mocked_ge, 'keys')
    assert all(key in mocked_ge.keys for key in [
        'learning_rate', 'optimizer', 'num_epochs',
        'batch_size', 'warmup_ratio', 'weight_decay'
    ])
    
    # Test punctuation
    assert hasattr(mocked_ge, 'punctuation')
    assert all(p in mocked_ge.punctuation for p in ['(', ')', '[', ']', ':', ','])

def test_remove_metrics(mocked_ge):
    """Test removal of evaluation metrics"""
    # Add some metrics
    mocked_ge.phenotype.update({
        'loss': 0.5,
        'accuracy': 0.8,
        'f1': 0.75,
        'precision': 0.7,
        'recall': 0.8
    })
    
    mocked_ge.remove_metrics()
    
    # Check metrics removed
    assert 'loss' not in mocked_ge.phenotype
    assert 'accuracy' not in mocked_ge.phenotype
    assert 'f1' not in mocked_ge.phenotype
    
    # Check essential keys remain
    assert 'learning_rate' in mocked_ge.phenotype
    assert 'optimizer' in mocked_ge.phenotype
    assert 'num_epochs' in mocked_ge.phenotype

def test_mutation(mocked_ge):
    """Test mutation operation"""
    original_genotype = mocked_ge.genotype.copy()
    
    mocked_ge.mutate()
    
    # Verify genotype changed
    assert len(mocked_ge.genotype) == len(original_genotype)
    diffs = [i for i, (a, b) in enumerate(zip(original_genotype, mocked_ge.genotype)) if a != b]
    assert len(diffs) > 0

def test_plus_minus_mutation(mocked_ge):
    """Test plus/minus mutation specifically"""
    original_genotype = mocked_ge.genotype.copy()
    
    mocked_ge.plus_minus_mutation()
    
    # Count changes
    changes = [(i, (old, new)) for i, (old, new) 
              in enumerate(zip(original_genotype, mocked_ge.genotype)) 
              if old != new]
    
    # Should have exactly one change
    assert len(changes) == 1
    
    # Change should be +1 or -1
    _, (old_val, new_val) = changes[0]
    assert abs(old_val - new_val) == 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])