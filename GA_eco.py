import random
import numpy as np
from typing import List, Any, Iterable
class GA:
    def __init__(self, shape, num_layers=3, num_nodes=8, mutation_rate=0.1, phenotype=None, genotype=None):
        self.shape = shape
        self.layers = num_layers
        self.nodes = num_nodes
        self.genotype = phenotype
        self.mutation_rate = mutation_rate
        self.activation_functions = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu',
                        'elu']
        self.optimizers = ['SGD', 'RMSprop', 'Adam',
                    'Adadelta', 'Adagrad', 'Adamax', 'Nadam']
        self.catch_phenotype(phenotype)
        return None
    
    def catch_phenotype(self, phenotype):
        if phenotype == None:
            self.phenotype = self.genotype_builder()
        else:
            self.phenotype = phenotype
        return self

    def genotype_builder(self):
        hidden_layers = int(3 * random.random()) + 1
        # The number of hidden layer nodes = 2/3 of the size of the input layer + the size of the output layer (1)
        nodes = int(20 * random.random()) + self.shape[1]
        afs = []
        #Activation function for all hidden layers +1 for output layer (this could be removed to have sigmoid(?))
        for _ in range(0, hidden_layers+2):
            af = self.activation_functions[random.randint(0, len(self.activation_functions)-1)]
            afs.append(af)

        optimiser = self.optimizers[random.randint(0, len(self.optimizers)-1)]
        epochs = int(abs(np.random.normal(50, 15)))
        batch_size = int(np.random.beta(2, 8))*10 + 1
        mutation_rate = round(np.random.beta(1, 7, 1)[0], 2)
        population_size = int(10 * random.random()) + 3
        cloning_rate = round(np.random.beta(4, 5, 1)[0], 2)
        max_generations = int(500 * random.random()) + 1
        phenotype = {
            'hidden layers' : hidden_layers,
            'nodes' : nodes,
            'activation functions' : afs,
            'optimiser' : optimiser,
            'number of epochs' : epochs,
            'batch size' : batch_size,
            'mutation rate' : mutation_rate,
            'population size' : population_size,
            'cloning rate' : cloning_rate,
            'max generations' : max_generations
        }
        return phenotype
    
    def remove_metrics(self):
        entries_to_remove = ('loss', 'accuracy', 'f1', 'precision', 'recall',
                              'mae', 'rmse', 'validation_accuracy', 'speed', 'val_acc_x_f1')
        if len(self.phenotype)-1 > 9:
            for k in entries_to_remove:
                self.phenotype.pop(k, None)
        return self.phenotype

    '''
    Deprecated function:
    '''
    def flatten_genotype(self):
        index = 0
        for _ in range(0, len(self.phenotype)-1):
            if isinstance(self.phenotype[index], list):
                for item in self.phenotype[index]:
                    self.phenotype.insert(index, item)
                    index += 1
            index += 1
        self.phenotype = [x for x in self.phenotype if not isinstance(x, list)]
        return None
    
    def rectify_phenotype(self, genotype=None):
        '''
        If the number of layers is greater than the number of activation functions in the phenotype,
        insert a random activation function to the 2nd to last index.

        Else if the number of layers is less than the number of activation functions in the phenotype,
        remove the 2nd to last activation function.
        '''
        while self.phenotype['hidden layers']+2 > len(self.phenotype['activation functions']):
            self.phenotype['activation functions'].insert(len(self.phenotype['activation functions'])-1, 
                                                        self.activation_functions[random.randint(0, len(self.activation_functions)-1)])
        while len(self.phenotype['activation functions']) > self.phenotype['hidden layers']+2:
            self.phenotype['activation functions'].remove(self.phenotype['activation functions'][len(self.phenotype['activation functions'])-1])
        return self

    def mutate(self):
        #stubborn gene idea = The last activation function has a further chance not to mutate if fitness is good?
        self.remove_metrics()
        which_mutation = random.choice(list(self.phenotype.items()))
        chance = random.randint(0, 100)
        if chance <= self.mutation_rate:
            if which_mutation[0] == 'hidden layers':
                self.phenotype[which_mutation[0]] = int(3 * random.random()) + 1
                if len(self.phenotype['activation functions']) > self.phenotype['hidden layers']+2:
                    del self.phenotype['activation functions'][1+self.phenotype['hidden layers']:-1]
                elif self.phenotype['hidden layers'] > len(self.phenotype['activation functions'])-2:
                    while self.phenotype['hidden layers'] > len(self.phenotype['activation functions'])-2:
                        self.phenotype['activation functions'].insert(len(self.phenotype['activation functions'])-1, self.activation_functions[random.randint(0, len(self.activation_functions)-1)])
            elif which_mutation[0] == 'nodes':
                self.phenotype[which_mutation[0]] = int(20 * random.random()) + self.shape[1]
            elif which_mutation[0] == 'activation functions':
                which_af = random.randint(0, len(self.phenotype[which_mutation[0]])-1)
                for i in range(0, len(self.phenotype['activation functions'])):
                    if which_af == i:
                        self.phenotype['activation functions'][i] = self.activation_functions[random.randint(0, len(self.activation_functions)-1)]
            elif which_mutation[0] == 'optimiser':
                self.phenotype[which_mutation[0]] = self.optimizers[random.randint(0, len(self.optimizers)-1)]
            elif which_mutation[0] == 'number of epochs':
                self.phenotype[which_mutation[0]] = int(abs(np.random.normal(50, 15)))
            elif which_mutation[0] == 'batch size':
                self.phenotype[which_mutation[0]] = int(np.random.beta(2, 8))*10 + 1
            elif which_mutation[0] == 'mutation rate':
                self.phenotype[which_mutation[0]] = round(np.random.beta(1, 7, 1)[0], 2)
            elif which_mutation[0] == 'population size':
                self.phenotype[which_mutation[0]] = int(20 * random.random()) + 3
            elif which_mutation[0] == 'cloning rate':
                self.phenotype[which_mutation[0]] = round(np.random.beta(4, 5, 1)[0], 2)
            elif which_mutation[0] == 'max generations':
                self.phenotype[which_mutation[0]] = int(500 * random.random()) + 1
        return self
    
    ###TESTING FUNCTION###
    def set_genotype(self, genotype):
        self.phenotype = genotype
        return self

if __name__ == '__main__':
    evo_parameters = {
        'mutation rate': 2,
        'population size': 10,
        'cloning rate': 0.33,
        'max generations': 3
    }
    individual = GA(shape=(536, 8), mutation_rate=100, evo_parameters=evo_parameters)
    #individual.mutate()
    #another_individual = GA(shape=8, mutation_rate=100)
    #two_children = individual.crossover(another_individual)
    #indi_geno = individual.remove_metrics()
    print (individual.phenotype)