import random
import numpy as np
from typing import List, Any, Iterable
class GE:
    def __init__(self, shape, num_layers=3, num_nodes=8, mutation_rate=0.1, genotype=None):
        self.shape = shape
        self.layers = num_layers
        self.nodes = num_nodes
        self.mutation_rate = mutation_rate
        self.grammar = self.get_grammar()
        if genotype == None:
            self.genotype = self.genotype_builder(32)
        else:
            self.genotype = genotype
        self.basic_ops = ['+', '/', '*', '-', ]
        self.keys = ['hidden layers', 'nodes', 'activation functions', 'optimiser', 'number of epochs', 'batch size']
        self.punctuation = ['(', ')', '[', '[\'', '], ', '\'], ', '\'] ', ']', ':', ',', '\"', '\'{', '}\'']
        self.phenotype = ""
        self.phenotype_builder()
        self.dictionise()
        self.n_layers = None
        return None
    
    def genotype_builder(self, genome_length):
        genome = random.sample(range(0, 40), genome_length)
        return genome
    
    @staticmethod
    def get_grammar():
        '''
        grammar = {
            'start' : [['expr']],

            'expr' : [['pre_op'], ['pre_op_non_term'],  ['pre_op', 'op', 'expr'], ['pre_op_non_term', 'op', 'expr'], ['input', 'op', 'expr'],
                    ['non_terminal', 'op', 'non_terminal'], ['(', 'non_terminal', 'op', 'non_terminal', ')'], ['(', 'input', 'op', 'expr', ')'],
                    ['(', 'pre_op', 'op', 'expr', ')'], ['(', 'pre_op_non_term', 'op', 'expr', ')']],

            'op' : [['+'], ['/'], ['*'], ['-']],
            
            'pre_op_non_term' : [['tf.math.sin', '(', 'non_terminal', ')'], ['tf.math.cos', '(', 'non_terminal', ')'],
            ['tf.math.tan', '(', 'non_terminal', ')'], ['tf.math.abs', '(', 'non_terminal', ')'],
            ['tf.math.minimum', '(', 'non_terminal', ',', 'terminal', ')'],
            ['tf.math.maximum', '(', 'non_terminal', ',', 'terminal', ')'],
            ['tf.math.reduce_max', '(', 'non_terminal', ')'], ['tf.math.tanh', '(', 'non_terminal', ')'],
            ['tf.math.square', '(', 'non_terminal', ')'], ['tf.math.sqrt', '(', 'non_terminal', ')'],
            ['tf.math.negative', '(', 'non_terminal', ')']], 

            'pre_op' : [['tf.math.sin', '(', 'input', ')'], ['tf.math.cos', '(', 'input', ')'],
            ['tf.math.tan', '(', 'input',')'], 
            ['tf.math.minimum', '(', 'input', ',', 'var', ')'],
            ['tf.math.maximum', '(', 'input', ',', 'var', ')'], ['tf.math.tanh', '(', 'input', ')'],
            ['tf.math.exp', '(', 'input', ')'], ['tf.math.reduce_sum', '(', 'input', ')'],],

            'non_terminal' : [['pre_op'], ['input']],
            'terminal' : [['var']],
            'input' : [['tensor']],
            'var' : [['0.1'], ['1.0'], ['2.0']]
        }
        '''

        grammar = {
            'start' : [['expr']],

            'expr' : [['\'{', '\"', 'hidden layers', '\"', ':', 'nodes_var', ',', '\"', 'nodes', '\"', ':', 'layers_var', ',', '\"', 'activation functions', '\"', ':', 
                       '[', '\"', 'activation_function', '\"', ',',  '\"', 'activation_function', '\"', ',', '\"', 'activation_function', '\"', ',', '\"', 'activation_function', '\"', ']', ',',
                        '\"', 'optimiser', '\"', ':', '\"', 'opti', '\"', ',', '\"', 'number of epochs', '\"', ':', 'epochs', ',', '\"', 'batch size', '\"', ':', 'batch_size', '}\'']],

            'activation_function' : [['acti_expr']],

            'acti_expr' : [['acti_pre_op'], ['acti_pre_op_non_term'],  ['acti_pre_op', 'op', 'acti_expr'], ['acti_pre_op_non_term', 'op', 'acti_expr'], ['acti_input', 'op', 'acti_expr'],
                    ['acti_non_terminal', 'op', 'acti_non_terminal'], ['(', 'acti_non_terminal', 'op', 'acti_non_terminal', ')'], ['(', 'acti_input', 'op', 'acti_expr', ')'], ['(', 'acti_pre_op', 'op', 'acti_expr', ')'], 
                    ['(', 'acti_pre_op_non_term', 'op', 'acti_expr', ')']],

            'acti_pre_op_non_term' : [['tf.math.sin', '(', 'acti_non_terminal', ')'], ['tf.math.cos', '(', 'acti_non_terminal', ')'],
            ['tf.math.tan', '(', 'acti_non_terminal', ')'], ['tf.math.abs', '(', 'acti_non_terminal', ')'],
            ['tf.math.minimum', '(', 'acti_input', ',', 'acti_non_terminal', ')'],
            ['tf.math.maximum', '(', 'acti_input', ',', 'acti_non_terminal', ')'], 
            ['tf.math.reduce_max', '(', 'acti_non_terminal', ')'], ['tf.math.tanh', '(', 'acti_non_terminal', ')'],
            ['tf.math.square', '(', 'acti_non_terminal', ')'], ['tf.math.sqrt', '(', 'acti_non_terminal', ')'],
            ['tf.math.negative', '(', 'acti_non_terminal', ')']], 

            'acti_pre_op' : [['tf.math.sin', '(', 'acti_input', ')'], ['tf.math.cos', '(', 'acti_input', ')'],
            ['tf.math.tan', '(', 'acti_input', ')'], 
            ['tf.math.minimum', '(', 'acti_input', ',', 'acti_var', ')'],
            ['tf.math.maximum', '(', 'acti_input', ',', 'acti_var', ')'], ['tf.math.exp', '(', 'acti_input', ')'],
            ['tf.math.reduce_sum', '(', 'acti_input', ')'], ['tf.math.tanh', '(', 'acti_input', ')']],

            'acti_non_terminal' : [['acti_input'], ['acti_pre_op']],

            'acti_input' : [['tensor']],

            'opti' : [['SGD'], ['RMSProp'], ['Adam'], ['Adadelta'], ['Adagrad'], ['Adamax'],
                    ['Nadam']],
            'learning_rate' : [['learn_var']],

            'batch_size' : [['batch_var']],

            'epochs' : [['epoch_var']],

            'op' : [['+'], ['/'], ['*'], ['-']],
            'nodes_var' : [['2'], ['4'], ['6'], ['8'], ['10']],
            'layers_var' : [['3']],
            'learn_var' : [['0.0001'], ['0.001'], ['0.1'], ['1.0']],
            'epoch_var' : [['5'], ['10'], ['20'], ['30'], ['40'], ['50']],
            'batch_var' : [['1'], ['4'], ['8'], ['16'], ['32']], #, ['64'], ['128'], ['256']
            'acti_var' : [['0.1'], ['1.0'], ['2.0']]
        }
        return grammar

    def phenotype_builder(self, genome=[], step=0, start_bool=True):
        returnable_pheno = ""
        if start_bool:
            try:
                self.phenotype == ""
            except (NameError):
                self.phenotype = ""
            step == 0
            codon = self.grammar['start'][self.genotype[step] % len(self.grammar['start'])][0]
            genome.append(codon)
            step += 1
            self.phenotype_builder(genome, step, False)
            ###
            #Reason for the return statements: "But returning a result from a function always returns
            #it to the direct caller of this function. It doesn't jump immediately out through several calls;"
        else:
            #Sanity check for wrapping
            if step == len(self.genotype):
                step = 0
            if len(genome) > 0:
                if (genome[0] in self.basic_ops or genome[0] in self.punctuation or genome[0] in self.keys):
                    self.phenotype = self.phenotype + str(genome[0])
                    genome.pop(0)
                    self.phenotype_builder(genome, step, False)
                    return None
                codon = self.grammar[genome[0]][self.genotype[step] % len(self.grammar[genome[0]])]
                #Non-Terminal                
                if codon[0] in self.grammar:  
                    genome.pop(0)
                    genome = codon + genome
                    step += 1
                    self.phenotype_builder(genome, step, False)
                    return None
                #Terminal check
                else:
                    self.phenotype = self.phenotype + str(codon[0])
                    genome.pop(0)
                    step += 1
                    
                    if len(codon) != 1:
                        t = codon[1:]
                        genome = t + genome
                    else:
                        codon = []
                    self.phenotype_builder(genome, step, False)
                    return None
        codon = []
        returnable_pheno = self.phenotype
        return returnable_pheno

    def remove_metrics(self):
        entries_to_remove = ('loss', 'accuracy', 'f1', 'precision', 'recall',
                              'mae', 'rmse', 'validation_accuracy', 'speed')
        if len(self.phenotype)-1 > 5:
            for k in entries_to_remove:
                self.phenotype.pop(k, None)
        return self.phenotype

    '''
    Deprecated function:
    '''
    def flatten_genotype(self):
        index = 0
        for _ in range(0, len(self.genotype)-1):
            if isinstance(self.genotype[index], list):
                for item in self.genotype[index]:
                    self.genotype.insert(index, item)
                    index += 1
            index += 1
        self.genotype = [x for x in self.genotype if not isinstance(x, list)]
        return None
    
    '''
    plus minus mutation - used for activation function mutation
    '''
    def mutate(self):
        self.remove_metrics()
        which_mutation = random.choice(range(0, len(self.genotype)-1))
        choices = [-1, 1]
        self.genotype[which_mutation] = self.genotype[which_mutation] + (random.choice(choices))
        return self
    
    def dictionise(self):
        import ast
        dict_pheno = ast.literal_eval(self.phenotype)
        result = ast.literal_eval(str(dict_pheno))
        self.phenotype = result
        return self

    ###TESTING FUNCTION###
    def set_genotype(self, genotype):
        self.genotype = genotype
        return self

if __name__ == '__main__':
    individual = GE(shape=8, mutation_rate=100)
    individual.pm_mutate()
    print (individual.phenotype)
   