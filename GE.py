import random
import numpy as np
import json
from typing import List, Any, Iterable
class GE:
    def __init__(self, shape, num_layers=4, num_nodes=8, mutation_rate=0.1, genotype=None, user_grammar_file=None,
                 genotype_length=32, gene_value=40):
        self.shape = shape
        self.layers = num_layers
        self.nodes = num_nodes
        self.mutation_rate = mutation_rate
        self.genotype_length = genotype_length
        self.gene_value = gene_value
        self.genes = 5
        self.set_grammar(user_grammar_file)
        self.catch_genotype(genotype=genotype)
        self.catch_grammar_exclusions()
        self.phenotype = ""
        self.build()
        return None
    
    def catch_genotype(self, genotype):
        '''
        Check to see if the genotype exists for this object, if not then create one with self.genotype_builder().
        
        Parameters:
            genotype (dictionary)/(None): Dictionary of the individuals genotype if exists, otherwise None.
        '''
        if genotype == None:
            self.genotype = self.genotype_builder(gene_value=self.gene_value, genotype_length=self.genotype_length)
        else:
            self.genotype = genotype
        return self

    def genotype_builder(self, gene_value, genotype_length):
        '''
        Build a random integer genotype with variable size and gene values.
        
        Paramaters:
            genotype_length (int): The maximum length of the genotype. (Default is 32)
            gene_value (int): The maximum value each gene can be. (Default=40)
            
        Returns:
            genome (list(int)): Random list of size genotype_length containing genes with a value betwee 0 to gene_value
        '''
        genome = random.sample(range(0, gene_value), genotype_length)
        return genome
    
    def catch_grammar_exclusions(self):
        self.basic_ops = ['+', '/', '*', '-', ]
        self.keys = ['hidden layers', 'nodes', 'activation functions', 'optimiser', 'number of epochs', 'batch size']
        self.punctuation = ['(', ')', '[', '[\'', '], ', '\'], ', '\'] ', ']', ':', ',', '\"', '\'{', '}\'']
        return self

    def build(self):
        '''
        This function catches recursive errors when building phenotypes. If a recursive error is found during 
        evolution then a poor phenotype will be produced, as an ode to a failed evolution (dead offspring).
        '''
        try:
            self.phenotype_builder()
            self.dictionise()
        except RecursionError:
            self.phenotype = {
                'hidden layers' : 1,
                'nodes' : 2,
                'activation functions' : ['tensor*0.0', 'tensor*0.0', 'tensor*0.0'],
                'optimiser' : 'Adam',
                'number of epochs' : 1,
                'batch size' : 8
            }
        return self
    
    def set_grammar(self, user_grammar_file=None):
        '''
        Set self.grammar, a default grammar is provided, however if the user has a grammar that can be inputted
        with the variable user_grammar_file.
        
        Parameters:
            user_grammar_file (str): File location of a JSON file containing the users grammar. Rules
            to create a grammar can be found in the .README.
        '''
        if user_grammar_file:
            g_file = open(user_grammar_file)
            jsonstr = g_file.read()
            json_dic = json.loads(jsonstr)
            self.grammar = json_dic
        else:
            self.grammar = {
                'start' : [['expr']],

                'expr' : [['\'{', '\"', 'hidden layers', '\"', ':', 'layers', ',', '\"', 'nodes', '\"', ':', 'n_nodes', ',', '\"', 'activation functions', '\"', ':', 
                        '[',  '\"', 'activation_function', '\"', ',',  '\"', 'activation_function', '\"',']', ',',
                            '\"', 'optimiser', '\"', ':', '\"', 'opti', '\"', ',', '\"', 'number of epochs', '\"', ':', 'epochs', ',', '\"', 'batch size', '\"', ':', 'batch_size', '}\'']],

                'activation_function' : [['acti_expr']],

                'acti_expr' : [['acti_pre_op'], ['acti_pre_op_non_term'],  ['acti_pre_op', 'op', 'acti_expr'], ['acti_pre_op_non_term', 'op', 'acti_expr'], ['acti_input', 'op', 'acti_expr'],
                        ['acti_non_terminal', 'op', 'acti_non_terminal'], ['(', 'acti_non_terminal', 'op', 'acti_non_terminal', ')'], ['(', 'acti_input', 'op', 'acti_expr', ')'], ['(', 'acti_pre_op', 'op', 'acti_expr', ')'], 
                        ['(', 'acti_pre_op_non_term', 'op', 'acti_expr', ')']],

                'acti_pre_op_non_term' : [['tf.math.sin', '(', 'acti_non_terminal', ')'], ['tf.math.cos', '(', 'acti_non_terminal', ')'],
                ['tf.math.tan', '(', 'acti_non_terminal', ')'], ['tf.math.abs', '(', 'acti_non_terminal', ')'],
                ['tf.math.minimum', '(', 'acti_input', ',', 'acti_non_terminal', ')'],
                ['tf.math.maximum', '(', 'acti_input', ',', 'acti_non_terminal', ')'], 
                ['tf.math.tanh', '(', 'acti_non_terminal', ')'],
                ['tf.math.square', '(', 'acti_non_terminal', ')'], ['tf.math.sqrt', '(', 'acti_non_terminal', ')'],
                ['tf.math.negative', '(', 'acti_non_terminal', ')']], 

                'acti_pre_op' : [['tf.math.sin', '(', 'acti_input', ')'], ['tf.math.cos', '(', 'acti_input', ')'],
                ['tf.math.tan', '(', 'acti_input', ')'], 
                ['tf.math.minimum', '(', 'acti_input', ',', 'acti_var', ')'],
                ['tf.math.maximum', '(', 'acti_input', ',', 'acti_var', ')'], ['tf.math.exp', '(', 'acti_input', ')'],
                ['tf.math.tanh', '(', 'acti_input', ')']],

                'acti_non_terminal' : [['acti_input'], ['acti_pre_op']],

                'acti_input' : [['tensor']],

                'opti' : [['SGD'], ['RMSProp'], ['Adam'], ['Adadelta'], ['Adagrad'], ['Adamax'],
                        ['Nadam']],

                'learning_rate' : [['learn_var']],

                'layers' : [['layers_var']],
                'n_nodes' : [['nodes_var']],
                'batch_size' : [['batch_var']],
                'epochs' : [['epoch_var']],

                'op' : [['+'], ['/'], ['*'], ['-']],
                'nodes_var' : [['2'], ['4'], ['6'], ['8'], ['10']], # This needs to incorporate the shape of the data, much like the GA nodes.
                'layers_var' : [['1'], ['2'], ['3']],
                'learn_var' : [['0.0001'], ['0.001'], ['0.1'], ['1.0']],
                'epoch_var' : [['5'], ['10'], ['15'], ['20'], ['25'], ['30'], ['40'], ['50']],
                'batch_var' : [['1'], ['1'], ['1'], ['1'], ['2'], ['2'], ['2'], ['3'], ['3'], ['4'], ['8'], ['16'], ['32'], ['64']], # ['128'], ['256']
                'acti_var' : [['0.1'], ['1.0'], ['2.0']]
            }
        return self

    def phenotype_builder(self, genotype=[], step=0, start_bool=True):
        '''
        A recursive function to build the individuals phenotype using self.genotype and self.grammar.

        Parameters:
            genotype (list)/(list(int)):
            step (int): The recursive step to parse the genotype. This variable is set to 0 if the 
                        recursive steps exceed the length of the genotype. IE a wrapping function.
            start_bool (bool): A boolean to set up variables such as self.phenotype and codon.

        Returns:
            returnable_pheno (str) : A phenotype which is equal to self.phenotype. This will be passed on
                                     to dictionise to become a dictionary.
        '''
        returnable_pheno = ""
        if start_bool:
            try:
                self.phenotype == ""
            except (NameError):
                self.phenotype = ""
            step == 0
            codon = self.grammar['start'][self.genotype[step] % len(self.grammar['start'])][0]
            genotype.append(codon)
            step += 1
            self.phenotype_builder(genotype, step, False)
        else:
            #Sanity check for wrapping
            if step == len(self.genotype):
                step = 0
            if len(genotype) > 0:
                if (genotype[0] in self.basic_ops or genotype[0] in self.punctuation or genotype[0] in self.keys):
                    self.phenotype = self.phenotype + str(genotype[0])
                    genotype.pop(0)
                    self.phenotype_builder(genotype, step, False)
                    return None
                codon = self.grammar[genotype[0]][self.genotype[step] % len(self.grammar[genotype[0]])]
                #Non-Terminal                
                if codon[0] in self.grammar:  
                    if codon[0] == 'nodes_var':
                        '''
                        self.phenotype[-3] is the number of hidden layers, this way we can dynamically change the grammar to include a new activation for each hidden layer
                        This section of code requires that NO change is made to the order of expr in the grammar.
                        '''
                        for i in range(int(self.phenotype[-10])):
                            symbols_to_add = ['\"', 'activation_function', '\"', ','] 
                            for j in range(0, len(symbols_to_add)):
                                self.grammar['expr'][0].insert(22+j, symbols_to_add[j])
                                genotype.insert(7+j, symbols_to_add[j])

                    genotype.pop(0)
                    genotype = codon + genotype
                    step += 1
                    self.phenotype_builder(genotype, step, False)
                    return None
                #Terminal check
                else:
                    self.phenotype = self.phenotype + str(codon[0])
                    genotype.pop(0)
                    step += 1
                    if len(codon) != 1:
                        t = codon[1:]
                        genotype = t + genotype
                    else:
                        codon = []
                    self.phenotype_builder(genotype, step, False)
                    return None
        codon = []
        returnable_pheno = self.phenotype
        self.set_grammar()
        return returnable_pheno


    def remove_metrics(self):
        '''
        A function to strip metrics from a phenotype, particularly used so we can use a variable number of
        activation functions in our phenotype and mutation and crossover continue to work as expected.
        
        Parameters:
            
        Returns:
            self.phenotype
        '''
        entries_to_remove = ('loss', 'accuracy', 'f1', 'precision', 'recall',
                              'mae', 'rmse', 'validation_accuracy', 'speed', 'val_acc_plus_f1')
        if len(self.phenotype)-1 > self.genes:
            for k in entries_to_remove:
                self.phenotype.pop(k, None)
        return self.phenotype
    
    def rectify_phenotype(self):
        '''
        Rebuild the phenotype. If the number of hidden layers have changed the phenotype has to be rebuilt to 
        incorporate more or less activation functions.
        
        '''

        self.phenotype = ""
        try:
            self.phenotype_builder()
            self.dictionise()
        except RecursionError:
            self.phenotype = {
                'hidden layers' : 1,
                'nodes' : 2,
                'activation functions' : ['tensor*0.0', 'tensor*0.0', 'tensor*0.0'],
                'optimiser' : 'Adam',
                'number of epochs' : 1,
                'batch size' : 8
            }
        
        return self
    
    def plus_minus_mutation(self):
        '''
        A plus minus mutation function. Metrics are stripped using self.remove_metrics and a random gene in the genotype
        is chosen for mutation, 1 will be subtracted or added to that gene. The phenotype will then be rebuilt.
        And the phenotype converted back to a dictionary.
        '''
        self.remove_metrics()
        which_mutation = random.choice(range(0, len(self.genotype)-1))
        choices = [-1, 1]
        self.genotype[which_mutation] = self.genotype[which_mutation] + (random.choice(choices))
        try:
            self.phenotype = ""
            self.phenotype_builder()
            self.dictionise()
        except RecursionError:
            self.phenotype = {
                'hidden layers' : 1,
                'nodes' : 2,
                'activation functions' : ['tensor*0.0', 'tensor*0.0', 'tensor*0.0'],
                'optimiser' : 'Adam',
                'number of epochs' : 1,
                'batch size' : 8
            }
        return self

    def mutate(self, type='pm'):
        if type == 'pm':
            self.plus_minus_mutation()
        return self
        
    
    def dictionise(self):
        '''
        Turn the phenotype from a string format into a dictionary using ast.literal_eval.
        Also, sets self.phenotype to the dictionary.
        '''
        import ast
        dict_pheno = ast.literal_eval(self.phenotype)
        result = ast.literal_eval(str(dict_pheno))
        self.phenotype = result
        return self

if __name__ == '__main__':
    individual = GE(shape=8, mutation_rate=100, user_grammar_file='basic_grammar.txt')
    print (individual.phenotype)
    
   