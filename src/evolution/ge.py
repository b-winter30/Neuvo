import random
import json
import ast
from typing import List, Dict, Optional, Union

class GE:
    """Grammatical Evolution implementation for transformer model evolution"""
    def __init__(self,
                 shape,
                 num_layers: int = 4,
                 num_nodes: int = 8,
                 mutation_rate: float = 0.1,
                 genotype: Optional[List[int]] = None,
                 user_grammar_file: Optional[str] = None,
                 genotype_length: int = 32,
                 gene_value: int = 40):
        
        self.shape = shape
        self.layers = num_layers
        self.nodes = num_nodes
        self.grammar_file = user_grammar_file
        self.mutation_rate = mutation_rate
        self.genotype_length = genotype_length
        self.gene_value = gene_value
        self.genes = 5
        
        # Initialize grammar and genotype
        self.set_grammar(user_grammar_file)
        self.catch_genotype(genotype=genotype)
        self.catch_grammar_exclusions()
        self.phenotype = ""
        self.build()

    def catch_genotype(self, genotype: Optional[List[int]]):
        """Initialize or use provided genotype"""
        if genotype is None:
            self.genotype = self.genotype_builder(
                gene_value=self.gene_value,
                genotype_length=self.genotype_length
            )
        else:
            self.genotype = genotype
        return self

    def genotype_builder(self, gene_value: int, genotype_length: int) -> List[int]:
        """Build random integer genotype"""
        return random.sample(range(0, gene_value), genotype_length)

    def catch_grammar_exclusions(self):
        """Define grammar tokens and operators"""
        self.basic_ops = ['+', '/', '*', '-']
        self.keys = ['learning_rate', 'optimizer', 'num_epochs', 'batch_size', 'warmup_ratio', 'weight_decay']
        self.punctuation = ['(', ')', '[', '[\'', '], ', '\'], ', '\'] ', ']', ':', ',', '\"', '\'{', '}\'']
        return self

    def set_grammar(self, user_grammar_file: Optional[str] = None):
        """Set grammar rules from file or use default transformer grammar"""
        if user_grammar_file:
            with open(user_grammar_file) as g_file:
                self.grammar = json.load(g_file)
        else:
            # Default grammar for transformer hyperparameters
            self.grammar = {
                'start': [['expr']],
                'expr': [['\'{',
                         '\"learning_rate\":', 'learning_rate', ',',
                         '\"optimizer\":', '\"', 'optimizer', '\"', ',',
                         '\"num_epochs\":', 'epochs', ',',
                         '\"batch_size\":', 'batch_size', ',',
                         '\"warmup_ratio\":', 'warmup_ratio', ',',
                         '\"weight_decay\":', 'weight_decay',
                         '}\'']],
                
                'learning_rate': [['1e-5'], ['2e-5'], ['3e-5'], ['4e-5'], ['5e-5']],
                'optimizer': [['AdamW'], ['Adam'], ['SGD'], ['RMSprop']],
                'epochs': [['2'], ['3'], ['4'], ['5']],
                'batch_size': [['4'], ['8'], ['16'], ['32']],
                'warmup_ratio': [['0.0'], ['0.1'], ['0.2']],
                'weight_decay': [['0.0'], ['0.01'], ['0.1']]
            }
        return self

    def build(self):
        """Build phenotype from genotype using grammar"""
        try:
            self.phenotype_builder()
            self.dictionise()
        except RecursionError:
            # Fallback phenotype if building fails
            self.phenotype = {
                'learning_rate': 2e-5,
                'optimizer': 'AdamW',
                'num_epochs': 3,
                'batch_size': 16,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01
            }
        return self

    def phenotype_builder(self, genotype: List = [], step: int = 0, start_bool: bool = True):
        """Recursively build phenotype using grammar rules"""
        if start_bool:
            if not hasattr(self, 'phenotype'):
                self.phenotype = ""
            step = 0
            codon = self.grammar['start'][self.genotype[step] % len(self.grammar['start'])][0]
            genotype.append(codon)
            step += 1
            self.phenotype_builder(genotype, step, False)
        else:
            if step == len(self.genotype):
                step = 0
            if len(genotype) > 0:
                current = genotype[0]
                if current in self.basic_ops or current in self.punctuation or current in self.keys:
                    self.phenotype += str(current)
                    genotype.pop(0)
                    self.phenotype_builder(genotype, step, False)
                    return None

                codon = self.grammar[genotype[0]][self.genotype[step] % len(self.grammar[genotype[0]])]
                
                if codon[0] in self.grammar:
                    genotype.pop(0)
                    genotype = codon + genotype
                    step += 1
                    self.phenotype_builder(genotype, step, False)
                    return None
                else:
                    self.phenotype += str(codon[0])
                    genotype.pop(0)
                    step += 1
                    if len(codon) > 1:
                        genotype = codon[1:] + genotype
                    self.phenotype_builder(genotype, step, False)
                    return None
        
        self.set_grammar(self.grammar_file)
        return self.phenotype

    def remove_metrics(self):
        """Remove evaluation metrics from phenotype"""
        entries_to_remove = ('loss', 'accuracy', 'f1', 'precision', 'recall',
                           'eval_loss', 'eval_accuracy', 'wer', 'bleu')
        if len(self.phenotype) - 1 > self.genes:
            for k in entries_to_remove:
                self.phenotype.pop(k, None)
        return self

    def mutate(self, type: str = 'pm'):
        """Mutate genotype"""
        if type == 'pm':
            self.plus_minus_mutation()
        return self

    def plus_minus_mutation(self):
        """Perform plus/minus mutation on genotype"""
        self.remove_metrics()
        which_mutation = random.randrange(len(self.genotype))
        self.genotype[which_mutation] += random.choice([-1, 1])
        
        try:
            self.phenotype = ""
            self.phenotype_builder()
            self.dictionise()
        except RecursionError:
            # Fallback phenotype if mutation produces invalid result
            self.phenotype = {
                'learning_rate': 2e-5,
                'optimizer': 'AdamW',
                'num_epochs': 3,
                'batch_size': 16,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01
            }
        return self

    def dictionise(self):
        """Convert phenotype string to dictionary"""
        dict_pheno = ast.literal_eval(self.phenotype)
        self.phenotype = ast.literal_eval(str(dict_pheno))
        return self