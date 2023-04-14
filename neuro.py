import pandas as pd
import time
import math
import random
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
from GA import GA
import warnings
import logging
warnings.filterwarnings('ignore') 
tf.get_logger().setLevel(logging.ERROR)
#from GE import GE
from Metrics import f1_m, precision_m, recall_m
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from sklearn.model_selection import StratifiedKFold
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import psutil

class Neuroevolution:
    def __init__(self, evo_params, dir_path, type, genotype=None):
        self.mutation_rate = self.parse_params(evo_params)['mutation_rate']
        self.population_size = self.parse_params(evo_params)['population_size']
        self.cloning_rate = self.parse_params(evo_params)['cloning_rate']
        self.max_generations = self.parse_params(evo_params)['max_generations']
        self.shape = self.load_data_return_shape(dir_path)
        self.type = type
        if self.type.lower() == 'ga':
            self.EA = GA(shape=self.shape, mutation_rate=self.mutation_rate, genotype=genotype)
        #elif self.type.lower() == 'ge':
        #    self.EA = GE(shape=self.shape, mutation_rate=self.mutation_rate)
        self.model = self.build_ann_custom_architecture()
        return None
    
    def build_basic_ann(self):
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(Dense(units=3, activation='softsign', input_dim=self.shape, use_bias=True))
        model.add(Dense(units=3, activation='softplus', use_bias=True))
        model.add(Dense(units=1, activation='sigmoid', use_bias=True))
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
        model.fit(self.dataX, self.dataY, batch_size=4, epochs=5, verbose=0, validation_data=(self.dataX, self.dataY), callbacks=[es])
        return None
    
    def build_ann_custom_architecture(self):
        tf.keras.backend.clear_session()
        layers, nodes, afs, optimiser, _, _ = list(self.EA.genotype.items())
        model = Sequential()
        model.add(Dense(units=nodes[1], activation=afs[1][0], input_dim=self.shape, use_bias=True))
        for i in range(layers[1]):
            model.add(Dense(units=nodes[1], activation=afs[1][i], use_bias=True))
        model.add(Dense(units=1, activation=afs[1][-1], use_bias=True))
        model.compile(optimizer=optimiser[1], loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m,
                        MeanAbsoluteError(), RootMeanSquaredError()])
        return model
    
    def run_ann(self):
        tf.keras.backend.clear_session()
        _, _, _, _, epochs, batch_size = list(self.EA.genotype.items())
        kfold = StratifiedKFold(n_splits=2, shuffle=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
        start = time.time()
        for i, (train_index, test_index) in enumerate(kfold.split(self.dataX, self.dataY)):
            X_train,X_test = self.dataX[train_index],self.dataX[test_index]
            Y_train,Y_test = self.dataY[train_index],self.dataY[test_index]
            self.model.fit(X_train, Y_train, batch_size=batch_size[1], epochs=epochs[1], verbose=0, validation_data=(X_test, Y_test), callbacks=[es])
            history = self.model.history.history
            last_val = history['val_accuracy'].pop()
        if last_val > 0.1:
            loss, accuracy, f1, precision, recall, mae, rmse = self.model.evaluate(X_test, Y_test, verbose=0)
        else:
            loss, accuracy, f1, precision, recall, mae, rmse = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if math.isnan(f1):
            f1 = 0
        end = time.time()
        speed = end - start
        metrics = {
            'loss' : loss,
            'accuracy' : accuracy,
            'f1' : f1,
            'precision' : precision,
            'recall' : recall,
            'mae' : mae,
            'rmse' : rmse,
            'validation_accuracy' : last_val,
            'speed' : speed
        }
        self.phenotype = dict(self.EA.genotype, **metrics)
        return None

    def parse_params(self, evo_param_file):
        params = {}
        with open(evo_param_file) as f:
            for line in f:
                (key, val) = line.split(' = ')
                try:
                    params[key] = int(val)
                except ValueError:
                    params[key] = float(val)
        return params
    
    def load_data_return_shape(self, dir_path):
        dataX = pd.read_csv(dir_path+'/x_data.csv', header=None)
        dataY = pd.read_csv(dir_path+'/y_data.csv', header=None)
        self.dataX = dataX.to_numpy()
        self.dataY = dataY.to_numpy()
        return self.dataX.shape[1]

    def evolve(self, evo_params):
        return None

    def build_ann_custom_af():
        return None
    
    def remove_metrics(self):
        entries_to_remove = ('loss', 'accuracy', 'f1', 'precision', 'recall',
                              'mae', 'rmse', 'validation_accuracy', 'speed')
        if len(self.phenotype)-1 > 5:
            for k in entries_to_remove:
                self.phenotype.pop(k, None)
        return self

    def sort_pop(self):
        print (self.population[0])
        sorted_pop = sorted(self.population.phenotype, key= lambda x: x['validation_accuracy'], reverse=True)
        print (sorted_pop)
        return self
    
    def flatten_phenotype(self):
        index = 0
        for key, value in self.phenotype:
            if isinstance(self.phenotype[key], list):
                for item in self.phenotype[key]:
                    self.phenotype.insert(index, item)
                    index += 1
            index += 1
        self.phenotype = [x for x in self.phenotype if not isinstance(x, list)]
        return self

class NeuroBuilder():
    def __init__(self, evo_params, dir_path, type):
        self.population_size = self.parse_params(evo_params)['population_size']
        self.parameter_file = evo_params
        self.max_generations = self.parse_params(evo_params)['max_generations']
        self.cloning_rate = self.parse_params(evo_params)['cloning_rate']
        self.mutation_rate = self.parse_params(evo_params)['mutation_rate']
        self.dir_path = dir_path
        self.type = type
        return None

    def parse_params(self, evo_param_file):
        params = {}
        with open(evo_param_file) as f:
            for line in f:
                (key, val) = line.split(' = ')
                try:
                    params[key] = int(val)
                except ValueError:
                    params[key] = float(val)
        return params
    
    def test(self):
        phenotype_list = []
        for individual in self.population:
            phenotype_list.append(individual.phenotype)
            #print(individual.phenotype)
        sorted_pop = sorted(phenotype_list, key= lambda x: x['validation_accuracy'], reverse=True)
        #print ('After sorting: ')
        for i in range(0, len(self.population)):
            self.population[i].phenotype = sorted_pop[i]
            #print(self.population[i].phenotype)
        #= sorted_pop
        return self

    def roulette_selection(self):
        import operator
        from collections import OrderedDict
        total_from_set = 0
        phenotype_list = []
        for individual in self.population:
            phenotype_list.append(individual.phenotype)

        sorted_pop = sorted(phenotype_list, key= lambda x: x['validation_accuracy'], reverse=True)
        for i in range(0, len(self.population)):
            self.population[i].phenotype = sorted_pop[i]

        n = math.ceil(len(self.population)*self.parse_params(self.parameter_file)['cloning_rate'])
        cloned_pop = self.population[:n]
        sorted_pop = self.population[n:]
        max_values = 0
        for i in range(len(sorted_pop)):
            max_values += sorted_pop[i].phenotype.get('validation_accuracy')
        
        choices = [0.0]
        for i in range(len(sorted_pop)):
            try:
                chance = (sorted_pop[i].phenotype.get('validation_accuracy') / max_values) * 100
            except (ZeroDivisionError):
                chance = 100 / len(sorted_pop)
            choices.append(choices[i] + chance)
        temp_pop = []
        while len(temp_pop) < len(sorted_pop):
            first_spinOfWheel = random.uniform(0, 100)
            second_spinOfWheel = random.uniform(0, 100)
            first_chosen_index = 0
            second_chosen_index = 0
            first_block = 0
            while first_chosen_index == 0:
                for i in range(len(choices)):
                    if first_spinOfWheel <= choices[i]:
                        first_chosen_index = i
                        first_block = choices[i-1]
                        break      
            same = True
            while second_chosen_index == 0 and same == True:
                for i in range(len(choices)):
                    if second_spinOfWheel <= choices[i]:
                        second_chosen_index = i
                        if second_chosen_index == first_chosen_index:
                            #second_spinOfWheel = random.uniform(0, 100)
                            second_spinOfWheel = random.choice([i for i in range(0, 100) if i not in range(int(first_block), int(first_spinOfWheel))])
                            second_chosen_index = i     
                        else:
                            same = False
                            break  
            child1, child2 = self.crossover(sorted_pop[first_chosen_index-1], sorted_pop[second_chosen_index-1])
            temp_pop.append(child1)
            temp_pop.append(child2)

        new_pop = []
        temp_pop = self.retrain_pop(temp_pop)
        new_pop.extend(cloned_pop)
        new_pop.extend(temp_pop)
        # print ('new pop: ')
        # for individual in new_pop:
        #     print (individual.phenotype)
        self.population = new_pop
        return self
    '''
    @staticmethod
    def remove_metrics(dictionary):
        entries_to_remove = ('loss', 'accuracy', 'f1', 'precision', 'recall',
                              'mae', 'rmse', 'validation_accuracy', 'speed')
        if len(dictionary)-1 > 5:
            for k in entries_to_remove:
                dictionary.pop(k, None)
        return dictionary
    '''

    def crossover(self, parent_one, parent_two):
        import copy
        parent_one = parent_one.remove_metrics()
        parent_two = parent_two.remove_metrics()
        child1 = parent_one
        child2 = parent_two
        crossover_point = random.randint(0, len(list(child1.phenotype.items()))-1)
        count = 0
        for key in child1.phenotype:
            if count <= crossover_point:
                # Dont swap the fitness metrics
                temp_value = child1.phenotype[key]
                child1.phenotype[key] = child2.phenotype[key]
                child2.phenotype[key] = temp_value
            count += 1
        return [child1, child2]
    
    def mutate(self):
        #stubborn gene idea = The last activation function has a further chance not to mutate if fitness is good?
        for individual in self.population:
            chance = random.randint(0, 100)
            if chance <= individual.mutation_rate:
                individual.EA.mutate()
                individual.run_ann()
        return self

    def retrain_pop(self, population):
        pop = []
        #self.population.clear()
        for individual in population:
            individual.run_ann()
            pop.append(individual)
        #self.population = pop
        return pop

    def initialise_pop(self):
        pop = []
        for _ in range(0, self.population_size):
            a = Neuroevolution(self.parameter_file, self.dir_path, type=self.type)
            a.run_ann()
            pop.append(a)
        self.population = pop
        return self
    
    def which_fittest(self):
        fittest_val = 0.0
        self.pop_average_fitness = 0.0
        for individual in self.population:
            #Testing catch - Hacky working around is the first if statement, there should be a 'validation accuracy'
            ###FIX THIS !!!!! Some phenotypes have not been run & check results, i think the crossover is retraining all population and ignoring the clones
            if 'validation_accuracy' not in individual.phenotype:
                individual.run_ann()
                print ('Individual after failed fitness check = ', individual.phenotype)

            if individual.phenotype['validation_accuracy'] >= fittest_val:
                self.fittest = individual
                fittest_val = individual.phenotype['validation_accuracy']
                self.pop_average_fitness += fittest_val
            
        self.pop_average_fitness = self.pop_average_fitness / len(self.population) 
        return self

def run():
    from rich.console import Console
    import gc
    console = Console()
    #This builds one individual
    data_loc = './Datasets/WisconsinCancer'
    dataset_name = data_loc.rsplit('/', 1)[-1]
    builder = NeuroBuilder('./evo_params.txt', './Datasets/WisconsinCancer', type='ga')
    builder.initialise_pop()
    max_generations = builder.max_generations
    cloning_rate = builder.cloning_rate
    mutation_rate = builder.mutation_rate
    population_size = builder.population_size
    #builder.test()
    #chance = random.randint(0, 100)
    # for individual in builder.population:
    #     print (individual.phenotype)
    # for i in range(0, max_generations):
    #     print ('Generation ', str(i), '...')
    #     builder.roulette_selection()
    #     individual.EA.mutate()
    # for individual in builder.population:
    #     print (individual.phenotype)

    plot_generation = []
    plot_best_fitness = []
    plot_avg_fitness = []
    output_file = dataset_name+'_GA_p_'+str(population_size)+'_mr_'+str(mutation_rate)+'_cr_'+str(cloning_rate)+'_mg'
    with console.status("[bold green]Running through generations...") as status:
        for i in range(max_generations+1):
            builder.roulette_selection()
            builder.mutate()
            builder.which_fittest()
            best_network = builder.fittest
            #Every 50th generation, save the fittest network in a file.
            if i % 50 == 0 or best_network.phenotype['validation_accuracy'] >= 1.0:
                num = str(i)
                string0 = "Validation accuracy = " + str(best_network.phenotype['validation_accuracy'])
                string00 = "Speed = " + str(best_network.phenotype['speed'])
                string1 = "MAE = " + str(best_network.phenotype['mae'])
                string2 = "Test acc = " + str(best_network.phenotype['accuracy'])
                string3 = "RMSE = " + str(best_network.phenotype['rmse'])
                string4 = "Precision = " + str(best_network.phenotype['precision'])
                try:
                    string5 = "Recall = " + str(best_network.phenotype['recall'])
                except ValueError:
                    string5 = "ROC AUC Error"
                string6 = "F Measure score = " + str(best_network.phenotype['f1'])
                string7 = "Individual = " + str(best_network.phenotype)
                with open('./Results/Reformatted/'+output_file+'.csv','a') as fd:
                    fd.write(num + "\n") 
                    fd.write(string0 + "\n")
                    fd.write(string00 + "\n")
                    fd.write(string1 + "\n")
                    fd.write(string2 + "\n")
                    fd.write(string3 + "\n")
                    fd.write(string4 + "\n")
                    fd.write(string5 + "\n")
                    fd.write(string6 + "\n")
                    fd.write(string7 + "\n")
                    fd.write("\n")
                    fd.close()
                if best_network.phenotype['validation_accuracy'] >= 1.0:
                    break
            plot_generation.append(i)
            plot_best_fitness.append(best_network.phenotype['validation_accuracy'])  
            plot_avg_fitness.append(builder.pop_average_fitness)  
            console.log(f"Generation {i} complete...")
        plt.plot(plot_generation, plot_best_fitness, label='Best fitness')
        plt.plot(plot_generation, plot_avg_fitness, label='Average fitness')
        plt.legend(loc='best')
        plt.xlabel('Number of Generations')
        plt.ylabel('Fitness measured in Validation Accuracy', size=12)
        i = 0
        filename = './Results/Reformatted/'+output_file
        while os.path.exists('{}{:d}.png'.format(filename, i)):
            i += 1
        plt.savefig('{}{:d}.png'.format(filename, i))
        plt.close()
        global highest
        highest = 0
    completion_message = 'Evolution complete'
    print (completion_message)
    gc.collect()
    return None

if __name__ == '__main__':
    for i in range(3,10):
        run()
        