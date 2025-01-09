import random
import numpy as np
from typing import Dict, List, Optional

class GA:
    """Genetic Algorithm implementation for transformer model evolution"""
    def __init__(self, 
                 shape, 
                 mutation_rate: float = 0.1, 
                 phenotype: Optional[Dict] = None,
                 genotype: Optional[Dict] = None,
                 eco: bool = False):
        self.shape = shape
        self.genotype = phenotype
        self.mutation_rate = mutation_rate
        
        # Define possible hyperparameters for transformer models
        self.learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
        self.optimizers = ['AdamW', 'Adam', 'SGD', 'RMSprop']
        self.batch_sizes = [4, 8, 16, 32]
        self.warmup_ratios = [0.0, 0.1, 0.2]
        self.weight_decays = [0.0, 0.01, 0.1]
        
        self.catch_eco(eco)
        self.catch_phenotype(phenotype)

    def catch_eco(self, eco: bool):
        """Setup ecological evolution parameters"""
        if eco:
            self.eco = True
            self.genes = 9  # Additional genes for eco mode
        else:
            self.eco = False
            self.genes = 5
        return self

    def catch_phenotype(self, phenotype: Optional[Dict]):
        """Initialize or use provided phenotype"""
        if phenotype is None:
            self.genotype = self.genotype_builder()
            self.phenotype = self.genotype
        else:
            self.phenotype = phenotype
        return self

    def genotype_builder(self) -> Dict:
        """Build a random genotype (hyperparameter set) for transformer models"""
        phenotype = {
            'learning_rate': random.choice(self.learning_rates),
            'optimizer': random.choice(self.optimizers),
            'num_epochs': int(abs(np.random.normal(3, 1))),
            'batch_size': random.choice(self.batch_sizes),
            'warmup_ratio': random.choice(self.warmup_ratios),
            'weight_decay': random.choice(self.weight_decays),
        }
        
        if self.eco:
            phenotype.update({
                'mutation_rate': round(np.random.beta(1, 7, 1)[0], 2),
                'population_size': int(10 * random.random()) + 3,
                'cloning_rate': round(np.random.beta(3, 4, 1)[0], 2),
                'max_generations': int(200 * random.random()) + 1
            })
        return phenotype

    def remove_metrics(self):
        """Remove evaluation metrics from phenotype"""
        entries_to_remove = ('loss', 'accuracy', 'f1', 'precision', 'recall',
                           'eval_loss', 'eval_accuracy', 'wer', 'bleu')
        if len(self.phenotype) - 1 > self.genes:
            for k in entries_to_remove:
                self.phenotype.pop(k, None)
        self.genotype = self.phenotype
        return self

    def mutate(self):
        """Mutate hyperparameters"""
        self.remove_metrics()
        
        # Select random hyperparameter to mutate
        which_mutation = random.choice(list(self.phenotype.items()))
        
        if random.random() <= self.mutation_rate:
            if which_mutation[0] == 'learning_rate':
                self.phenotype[which_mutation[0]] = random.choice(self.learning_rates)
            elif which_mutation[0] == 'optimizer':
                self.phenotype[which_mutation[0]] = random.choice(self.optimizers)
            elif which_mutation[0] == 'num_epochs':
                self.phenotype[which_mutation[0]] = max(1, int(abs(np.random.normal(3, 1))))
            elif which_mutation[0] == 'batch_size':
                self.phenotype[which_mutation[0]] = random.choice(self.batch_sizes)
            elif which_mutation[0] == 'warmup_ratio':
                self.phenotype[which_mutation[0]] = random.choice(self.warmup_ratios)
            elif which_mutation[0] == 'weight_decay':
                self.phenotype[which_mutation[0]] = random.choice(self.weight_decays)
                
            if self.eco:
                if which_mutation[0] == 'mutation_rate':
                    self.phenotype[which_mutation[0]] = round(np.random.beta(1, 7, 1)[0], 2)
                elif which_mutation[0] == 'population_size':
                    self.phenotype[which_mutation[0]] = int(10 * random.random()) + 3
                elif which_mutation[0] == 'cloning_rate':
                    self.phenotype[which_mutation[0]] = round(np.random.beta(3, 4, 1)[0], 2)
                elif which_mutation[0] == 'max_generations':
                    self.phenotype[which_mutation[0]] = int(200 * random.random()) + 1
                    
        self.genotype = self.phenotype
        return self