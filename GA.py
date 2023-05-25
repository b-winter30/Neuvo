import random
import numpy as np
from typing import List, Any, Iterable
class GA:
    def __init__(self, shape, num_layers=3, num_nodes=8, mutation_rate=0.1, phenotype=None):
        self.shape = shape
        self.layers = num_layers
        self.nodes = num_nodes
        self.mutation_rate = mutation_rate
        self.activation_functions = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu',
                        'elu']
        self.optimizers = ['SGD', 'RMSprop', 'Adam',
                    'Adadelta', 'Adagrad', 'Adamax', 'Nadam']
        if phenotype == None:
            self.phenotype = self.genome_builder()
        else:
            self.phenotype = phenotype
        return None
    
    def genome_builder(self):
        hidden_layers = int(3 * random.random()) + 1
        # The number of hidden layer nodes = 2/3 of the size of the input layer + the size of the output layer (1)
        nodes = int(40 * random.random()) + self.shape[1]
        afs = []
        #Activation function for all hidden layers +1 for output layer (this could be removed to have sigmoid(?))
        for i in range(0, hidden_layers+1):
            af = self.activation_functions[random.randint(0, len(self.activation_functions)-1)]
            afs.append(af)

        optimiser = self.optimizers[random.randint(0, len(self.optimizers)-1)]
        epochs = int(50 * random.random()) + 1
        batch_size = int(64 * random.random()) + 1
        phenotype = {
            'hidden layers' : hidden_layers,
            'nodes' : nodes,
            'activation functions' : afs,
            'optimiser' : optimiser,
            'number of epochs' : epochs,
            'batch size' : batch_size
        }
        #genotype = [hidden_layers, nodes, afs, optimiser, epochs, batch_size]
        #self.phenotype = genotype
        return phenotype
    
    def remove_metrics(self):
        entries_to_remove = ('loss', 'accuracy', 'f1', 'precision', 'recall',
                              'mae', 'rmse', 'validation_accuracy', 'speed', 'val_acc_plus_f1')
        if len(self.phenotype)-1 > 5:
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
    
    def mutate(self):
        #stubborn gene idea = The last activation function has a further chance not to mutate if fitness is good?
        self.remove_metrics()
        which_mutation = random.choice(list(self.phenotype.items()))
        chance = random.randint(0, 100)
        if chance <= self.mutation_rate:
            if which_mutation[0] == 'hidden layers':
                self.phenotype[which_mutation[0]] = int(3 * random.random()) + 1
            elif which_mutation[0] == 'nodes':
                self.phenotype[which_mutation[0]] = int(40 * random.random()) + self.shape[1]
            elif which_mutation[0] == 'activation functions':
                which_af = random.randint(0, len(self.phenotype[which_mutation[0]])-1)
                for i in range(0, len(self.phenotype[which_mutation[0]])-1):
                    if which_af == i:
                        self.phenotype[which_mutation[0]][i] = self.activation_functions[random.randint(0, len(self.activation_functions)-1)]
            elif which_mutation[0] == 'optimiser':
                self.phenotype[which_mutation[0]] = self.optimizers[random.randint(0, len(self.optimizers)-1)]
            elif which_mutation[0] == 'number of epochs':
                self.phenotype[which_mutation[0]] = int(50 * random.random()) + 1
            elif which_mutation[0] == 'batch size':
                self.phenotype[which_mutation[0]] = int(64 * random.random()) + 1
        return self
    
    ###TESTING FUNCTION###
    def set_genotype(self, genotype):
        self.phenotype = genotype
        return self

if __name__ == '__main__':
    individual = GA(shape=8, mutation_rate=100)
    individual.mutate()
    another_individual = GA(shape=8, mutation_rate=100)
    two_children = individual.crossover(another_individual)
    indi_geno = individual.remove_metrics()
    print (indi_geno)