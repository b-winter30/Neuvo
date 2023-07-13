import pandas as pd
import time
import math
import random
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
from rich.console import Console
import copy
import gc
from GA import GA
import warnings
import logging
warnings.filterwarnings('ignore') 
tf.get_logger().setLevel(logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from Metrics import f1_m, precision_m, recall_m
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError, Precision, Recall
from sklearn.model_selection import StratifiedKFold
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from GE import GE

class Neuroevolution:
    def __init__(self, data, type, evo_params=None, genotype=None, fittest=None,  
                 eco=False, verbose=0, genotype_length=32, gene_value=40, grammar_file=None):
        self.parameter_list = evo_params
        self.eco=eco
        self.fittest = fittest
        self.catch_eco()
        self.string = ""
        self.type = type
        self.grammar_file=grammar_file
        self.verbose = verbose
        self.dataX = data[0]
        self.dataY = data[1]
        self.shape = self.dataX.shape
        self.genotype = genotype
        self.build_parent(genotype_length, gene_value)
        return None

    def build_architecture(self):
        if len(self.shape) > 2:
            self.model = self.build_cnn_custom_architecture()
        else:
            self.model = self.build_ann_custom_architecture()
        return self
    
    def build_ea(self, genotype_length=32, gene_value=40):
        if self.type.lower() == 'ga':
            self.EA = GA(shape=self.shape, mutation_rate=self.mutation_rate, phenotype=self.genotype, eco=self.eco)
        else:
            self.EA = GE(shape=self.shape, mutation_rate=self.mutation_rate, genotype=self.genotype,
                        genotype_length=genotype_length, gene_value=gene_value, user_grammar_file=self.grammar_file)

    def build_parent(self, genotype_length=32, gene_value=40):
        self.build_ea(genotype_length=genotype_length, gene_value=gene_value)
        self.build_architecture()
        return self

    def catch_eco(self):
        '''
        A function to check whether eco mode is activated, if it is then the evolutionary parameters will be
        set by the fittest individuals genes, else it will be chosen by the user from a parameter_list or 
        inputted at runtime by the user.
        '''
        if self.fittest != None:
            self.mutation_rate = self.fittest['mutation rate']
            self.population_size = self.fittest['population size']
            self.cloning_rate = self.fittest['cloning rate']
            self.max_generations = self.fittest['max generations']
        else:
            self.mutation_rate = self.parameter_list['mutation rate']
            self.population_size = self.parameter_list['population size']
            self.cloning_rate = self.parameter_list['cloning rate']
            self.max_generations = self.parameter_list['max generations']
        return self

    def build_basic_ann(self):
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(Dense(units=8, activation='relu', input_dim=self.shape[1], use_bias=True))
        model.add(Dense(units=8, activation='relu', use_bias=True))
        model.add(Dense(units=8, activation='relu', use_bias=True))
        model.add(Dense(units=8, activation='relu', use_bias=True))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        num_folds = 2
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3)
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True)
        start = time.time()
        loss, accuracy, f1, precision, recall, mae, rmse = (0.0,)*7
        for i, (train_index, test_index) in enumerate(kfold.split(self.dataX, self.dataY)):
            X_train,X_test = self.dataX[train_index],self.dataX[test_index]
            Y_train,Y_test = self.dataY[train_index],self.dataY[test_index]
            self.model.fit(X_train, Y_train, batch_size=4, epochs=20,
                            verbose=self.verbose, validation_data=(X_test, Y_test), callbacks=[es, TerminateOnNaN()])
            history = self.model.history.history
            last_val = history['val_accuracy'].pop()
            los, acc, f, prec, rec, ma, rms = self.model.evaluate(X_test, Y_test, verbose=self.verbose)
            loss += los
            accuracy += acc
            f1 += f
            precision += prec
            recall += rec
            mae += ma
            rmse += rms
        loss, accuracy, f1, precision, recall, mae, rmse = (loss/num_folds), (accuracy/num_folds), (f1/num_folds), \
            (precision/num_folds), (recall/num_folds), (mae/num_folds), (rmse/num_folds)
        end = time.time()
        speed = end - start
        val_acc_x_f1 = last_val * f1
        if math.isnan(val_acc_x_f1):
            val_acc_x_f1 = 0.0
        metrics = {
            'loss' : loss,
            'accuracy' : accuracy,
            'f1' : f1,
            'precision' : precision,
            'recall' : recall,
            'mae' : mae,
            'rmse' : rmse,
            'validation_accuracy' : last_val,
            'speed' : speed,
            'val_acc_x_f1' : val_acc_x_f1
        }
        return metrics
    
    def custom(self, tensor):
        sub_string = self.string
        x = eval(sub_string)
        return x
    
    def build_ann_custom_architecture(self):
        tf.keras.backend.clear_session()
        model = Sequential()
        try:
            model.add(Dense(units=self.EA.phenotype['nodes'], activation=str(self.EA.phenotype['activation functions'][0]), input_dim=self.shape[1], use_bias=True))
            for i in range(1, self.EA.phenotype['hidden layers']):
                model.add(Dense(units=self.EA.phenotype['nodes'], activation=str(self.EA.phenotype['activation functions'][i]), use_bias=True))
            model.add(Dense(units=self.dataY.shape[-1], activation=str(self.EA.phenotype['activation functions'][-1])))
            model.compile(optimizer=self.EA.phenotype['optimiser'], loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m,
                            MeanAbsoluteError(), RootMeanSquaredError()])
        except ValueError:
            self.string = self.EA.phenotype['activation functions'][0]
            get_custom_objects().update({'custom': self.custom})
            model.add(Dense(units=self.EA.phenotype['nodes'], activation=self.custom, input_dim=self.shape[1], use_bias=True))
            for i in range(1, self.EA.phenotype['hidden layers']):
                self.string = self.EA.phenotype['activation functions'][i]
                model.add(Dense(units=self.EA.phenotype['nodes'], activation=self.custom, use_bias=True))
            self.string = self.EA.phenotype['activation functions'][-1]
            model.add(Dense(units=self.dataY.shape[-1], activation=self.custom))
            model.compile(optimizer=self.EA.phenotype['optimiser'], loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m,
                            MeanAbsoluteError(), RootMeanSquaredError()])
        return model
    
    def run_ann(self, queue=None):
        tf.keras.backend.clear_session()
        num_folds = 2
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3)
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True)
        start = time.time()
        loss, accuracy, f1, precision, recall, mae, rmse = (0.0,)*7
        for i, (train_index, test_index) in enumerate(kfold.split(self.dataX, self.dataY)):
            X_train,X_test = self.dataX[train_index],self.dataX[test_index]
            Y_train,Y_test = self.dataY[train_index],self.dataY[test_index]
            self.model.fit(X_train, Y_train, batch_size=self.EA.phenotype['batch size'], epochs=self.EA.phenotype['number of epochs'],
                            verbose=self.verbose, validation_data=(X_test, Y_test), callbacks=[es, TerminateOnNaN()])
            history = self.model.history.history
            last_val = history['val_accuracy'].pop()
            los, acc, f, prec, rec, ma, rms = self.model.evaluate(X_test, Y_test, verbose=self.verbose)
            loss += los
            accuracy += acc
            f1 += f
            precision += prec
            recall += rec
            mae += ma
            rmse += rms
        loss, accuracy, f1, precision, recall, mae, rmse = (loss/num_folds), (accuracy/num_folds), (f1/num_folds), \
            (precision/num_folds), (recall/num_folds), (mae/num_folds), (rmse/num_folds)
        end = time.time()
        speed = end - start
        val_acc_x_f1 = last_val * f1
        if math.isnan(val_acc_x_f1):
            val_acc_x_f1 = 0.0
        metrics = {
            'loss' : loss,
            'accuracy' : accuracy,
            'f1' : f1,
            'precision' : precision,
            'recall' : recall,
            'mae' : mae,
            'rmse' : rmse,
            'validation_accuracy' : last_val,
            'speed' : speed,
            'val_acc_x_f1' : val_acc_x_f1
        }
        self.EA.phenotype = dict(self.EA.phenotype, **metrics)
        self.phenotype = self.EA.phenotype
        if self.type == 'ga':
            self.EA.genotype = self.EA.phenotype
        return None
    
    def build_cnn_custom_architecture(self):
        tf.keras.backend.clear_session()
        model = Sequential()
        try:
            model.add(Conv2D(self.EA.phenotype['nodes'], kernel_size=(3, 3), activation=self.EA.phenotype['activation functions'][0], input_shape=self.dataX.shape[1:]))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(self.EA.phenotype['nodes']*2, kernel_size=(3, 3), activation=self.EA.phenotype['activation functions'][0]))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            for i in range(1, self.EA.phenotype['hidden layers']):
                self.string = self.EA.phenotype['activation functions'][i]
                model.add(Dense(self.EA.phenotype['nodes']*4, activation=self.EA.phenotype['activation functions'][i]))
            model.add(Dropout(0.2))
            model.add(Dense(self.dataY.shape[-1], activation=self.EA.phenotype['activation functions'][-1]))
            
            model.compile(optimizer=str(self.EA.phenotype['optimiser']), loss=tf.keras.losses.Hinge(),
                       metrics=['accuracy', Precision(), Recall(), MeanAbsoluteError(), RootMeanSquaredError()])
        except ValueError:
            self.string = self.EA.phenotype['activation functions'][0]
            get_custom_objects().update({'custom': self.custom})
            shape = self.dataX.shape[1:]
            model.add(tf.keras.layers.Conv2D(self.EA.phenotype['nodes']/2, (3, 3), activation=self.custom, input_shape=shape))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            for i in range(1, self.EA.phenotype['hidden layers']):
                self.string = self.EA.phenotype['activation functions'][i]
                model.add(tf.keras.layers.Conv2D(self.EA.phenotype['nodes'], (3, 3), activation=self.custom))
                model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.string = self.EA.phenotype['activation functions'][-1]
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(self.EA.phenotype['nodes'], activation=self.EA.phenotype['activation functions'][-2]))
            model.add(tf.keras.layers.Dense(self.dataY.shape[-1], activation=self.EA.phenotype['activation functions'][-1]))
            model.compile(optimizer=str(self.EA.phenotype['optimiser']), loss=tf.keras.losses.Hinge(),
                        metrics=['accuracy', Precision(), Recall(), MeanAbsoluteError(), RootMeanSquaredError()])
        return model

    def run_cnn(self, queue=None):
        tf.keras.backend.clear_session()
        num_folds = 2
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=2)
        start = time.time()
        loss, accuracy, f1, precision, recall, mae, rmse = (0.0,)*7
        for i, (train_index, test_index) in enumerate(kfold.split(self.dataX, self.dataY)):
            X_train,X_test = self.dataX[train_index],self.dataX[test_index]
            Y_train,Y_test = self.dataY[train_index],self.dataY[test_index]
            self.model.fit(X_train, Y_train, batch_size=self.EA.phenotype['batch size'], epochs=self.EA.phenotype['number of epochs'], 
                           verbose=self.verbose, validation_data=(X_test, Y_test), callbacks=[es, TerminateOnNaN()],
                           use_multiprocessing=True)
            history = self.model.history.history
            last_val = history['val_accuracy'].pop()
            los, acc, prec, rec, ma, rms = self.model.evaluate(X_test, Y_test, verbose=self.verbose)
            loss += los
            accuracy += acc
            f1 += 2*(rec*prec)/(rec+prec)
            precision += prec
            recall += rec
            mae += ma
            rmse += rms
        loss, accuracy, f1, precision, recall, mae, rmse = (loss/num_folds), (accuracy/num_folds), (f1/num_folds), \
            (precision/num_folds), (recall/num_folds), (mae/num_folds), (rmse/num_folds)
        end = time.time()
        speed = end - start
        val_acc_x_f1 = last_val * f1
        if math.isnan(val_acc_x_f1):
            val_acc_x_f1 = 0.0
        metrics = {
            'loss' : loss,
            'accuracy' : accuracy,
            'f1' : f1,
            'precision' : precision,
            'recall' : recall,
            'mae' : mae,
            'rmse' : rmse,
            'validation_accuracy' : last_val,
            'speed' : speed,
            'val_acc_x_f1' : val_acc_x_f1
        }
        self.EA.phenotype = dict(self.EA.phenotype, **metrics)
        if self.type == 'ga':
            self.EA.genotype = self.EA.phenotype
        self.phenotype = self.EA.phenotype
        #queue.put(self.phenotype)
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

class NeuvoBuilder():
    def __init__(self, evo_params=None, type='ga', fittest=None, eco=False, verbose=0, gene_value=40, genotype_length=32,
                 fitness_function='f1', grammar_file=None):
        self.fitness_function = fitness_function
        self.fittest = fittest
        self.type = type
        self.grammar_file=grammar_file
        self.dataset_name = ""
        self.gene_value = gene_value
        self.genotype_length = genotype_length
        self.verbose = verbose
        self.eco = eco
        self.initialise_empty()
        self.catch_eco()
        self.check_possibility()
        return None
    
    def help(self):
        ''''
        self.fitness_function (str) : The user defined fitness function. Possible fitness functions include:
                                      ['loss', 'accuracy', 'f1', 'precision', 'recall', 'mae', 'rmse',
                                       'validation_accuracy', 'speed', 'val_acc_x_f1']
        self.fittest (Neuroevolution obj) : The fittest individual of the current generation.
        self.type (str) : Users decision on which evolutionary algorithm they want to use.
        self.dataset_name (str) : The name of the dataset the user has inputted. This variable is used when outputting
                                  results to a file.
        self.gene_value (int) : The maximum value for a gene in the grammatical evolutions genotype.
        self.genotype_length (int) : The maximum length of the genotype in the grammatical evolutions genotype.
        self.verbose (int) : A set function to set tensorflows verbose to a value, if verbose is 1 or 2, users
                             will be able to see the training of their networks.
        self.eco (bool) : Users decision to set eco mode to True or False. If true, the phenotypes created will include
                          evolutionary parameters. The evolutionary parameters of the elite individual will be used to 
                          control the evolution of the whole population.
        '''
        return None

    def initialise_empty(self):
        '''
        Initialiser function to set the evolutionary parameters to None. This function is needed
        to allow eco mode to work.
        '''
        self.mutation_rate = None
        self.population_size = None
        self.cloning_rate = None
        self.max_generations = None
        return self
        
    def check_possibility(self):
        '''
        A function to send error messages for non-implemented combinations of the framework.

        Current non-implemented combinations:
            - eco
            - GE AND eco
        '''
        if self.eco == True:
            not_possible = True
        else:
            not_possible = False
        assert not_possible == False, 'GE eco mode has not been implemented yet.'
        return self

    def catch_eco(self):
        '''
        A function to check whether eco mode is activated, if it is then the evolutionary parameters will be
        set by the fittest individuals genes, else it will be chosen by the user from a parameter_file or 
        inputted at runtime by the user.
        '''
        if self.eco:
            self.parameter_list = {'mutation rate' : self.fittest.EA.phenotype['mutation rate'], 
                                   'population size' : self.fittest.EA.phenotype['population size'], 
                                   'cloning rate' : self.fittest.EA.phenotype['cloning rate'],
                                   'max generations' : self.fittest.EA.phenotype['max generations']}
        else:
            self.parameter_list = {'mutation rate' : self.mutation_rate, 
                                   'population size' : self.population_size, 
                                   'cloning rate' : self.cloning_rate,
                                   'max generations' : self.max_generations}
        return self
    
    def parse_params(self, evo_param_file):
        '''
        A function to read the evolutionary parameters from a file instead of inputting them.
        The file should be a .txt file, an example in the following format is below.

        mutation_rate = 0.01
        cloning_rate = 0.3
        max_generations = 500
        population_size = 10

        Parameters:
            evo_param_file (string) : The name of the users evolutionary parameter file.

        Returns:
            params (dict) : A dictionary containing the users evolutionary parameters.
        '''
        params = {}
        with open(evo_param_file) as f:
            for line in f:
                (key, val) = line.split(' = ')
                try:
                    params[key] = int(val)
                except ValueError:
                    params[key] = float(val)
        return params

    def tournament_selection(self, tournament_size=2):
        '''

        '''
        assert tournament_size < len(self.population), "Tournament size must be less than or equal to the size of the population."
        retrain_pop = []
        population_copy = copy.copy(self.population)
        cloned_pop = []
        
        if self.elite_mode:
            cloned_pop.append(self.fittest)
        n = math.ceil(len(population_copy)*self.parameter_list['cloning rate'])-len(cloned_pop)

        cloned_pop.extend(population_copy[:n])
        reproducible_pop = population_copy[n:]
        j = len(retrain_pop)
        while j < self.population_size-len(cloned_pop):
            random_choices = random.sample(reproducible_pop, tournament_size)
            child1, child2 = self.crossover(random_choices[0], random_choices[1])
            retrain_pop.append(child1)
            j += 1
            retrain_pop.append(child2)
            j += 1
        new_pop = []
        temp_pop = retrain_pop
        temp_pop = self.retrain_pop(retrain_pop)
        
        new_pop.extend(cloned_pop)
        new_pop.extend(temp_pop)
        self.population = new_pop
        return self

    def roulette_selection(self):
        '''

        '''
        cloned_pop = []
        if self.elite_mode:
            cloned_pop.append(self.fittest)
        population_copy = copy.copy(self.population)
        phenotype_list = []
        for individual in population_copy:
            phenotype_list.append(individual.EA.phenotype)

        sorted_phenotypes = sorted(phenotype_list, key= lambda x: x[self.fitness_function], reverse=True)
        for i in range(0, len(population_copy)):
            population_copy[i].EA.phenotype = sorted_phenotypes[i]

        n = math.ceil(len(population_copy)*self.parameter_list['cloning rate'])-len(cloned_pop)
        cloned_pop.extend(population_copy[:n])
        sorted_phenotypes = population_copy[n:]
        max_values = 0
        for i in range(len(sorted_phenotypes)):
            max_values += sorted_phenotypes[i].EA.phenotype.get(self.fitness_function)
        
        choices = [0.0]
        for i in range(len(sorted_phenotypes)):
            try:
                chance = (sorted_phenotypes[i].EA.phenotype.get(self.fitness_function) / max_values) * 100
            except (ZeroDivisionError):
                chance = 100 / len(sorted_phenotypes)
            choices.append(choices[i] + chance)
        temp_pop = []
        while len(temp_pop) < len(sorted_phenotypes):
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
                            second_spinOfWheel = random.choice([i for i in range(0, 100) if i not in range(int(first_block), int(first_spinOfWheel))])
                            second_chosen_index = i     
                        else:
                            same = False
                            break  
            child1, child2 = self.crossover(sorted_phenotypes[first_chosen_index-1], sorted_phenotypes[second_chosen_index-1])
            temp_pop.append(child1)
            temp_pop.append(child2)

        new_pop = []
        temp_pop = self.retrain_pop(temp_pop)
        new_pop.extend(cloned_pop)
        new_pop.extend(temp_pop)
        self.population = new_pop
        return self

    def crossover(self, parent_one, parent_two):
        '''
        Parent crossover function, used to find out which evolutionary algorithm has been chosen
        and use crossover_ge or crossover_ga accordingly. Metrics are also removed before from the
        individuals phenotype before crossover.

        Parameters:
            parent_one (Neuroevolution obj) : The first parent chosen for reproduction.
            parent_two (Neuroevolution obj) : The second parent chosen for reproduction.
        
        Returns:
            children list[Neuroevolution obj, Neuroevolution obj] : A list containing two Neuroevolution
                                                           objects. These will then be passed to
                                                           retrain.
        '''
        parent_one.EA.remove_metrics()
        parent_two.EA.remove_metrics()
        if self.type == 'ge':
            children = self.crossover_ge(parent_one=parent_one, parent_two=parent_two)
        elif self.type == 'ga':
            children = self.crossover_ga(parent_one=parent_one, parent_two=parent_two)
        return children
    
    def crossover_ga(self, parent_one, parent_two):
        '''
        The crossover function for GA. This is the same function as it is for GE,
        However, determining the crossover point is different, as opposed to GE, 
        in GA the genotype is the same as the phenotype.
        
        Here we determine a random gene in the genotype to be the 'slice' or 
        crossover point in which the genes between 0 and the crossover point
        from both parents will be recombinated and placed into two offspring.
        
        Parameters:
            parent_one (Neuroevolution obj) : The first parent chosen for reproduction.
            parent_two (Neuroevolution obj) : The second parent chosen for reproduction.
        
        Returns:
            list[Neuroevolution obj, Neuroevolution obj] : A list containing two Neuroevolution
                                                           objects. These will then be passed to
                                                           retrain.
        '''
        
        child1 = parent_one
        child2 = parent_two
        crossover_point = random.randint(0, len(list(child1.EA.phenotype.items()))-1)
        count = 0
        for key in child1.EA.phenotype:
            if count <= crossover_point:
                # Dont swap the fitness metrics
                temp_value = child1.EA.phenotype[key]
                child1.EA.phenotype[key] = child2.EA.phenotype[key]
                child2.EA.phenotype[key] = temp_value
            count += 1
        child1.EA.rectify_phenotype()
        child2.EA.rectify_phenotype()
        child1.phenotype = child1.EA.phenotype
        child2.phenotype = child2.EA.phenotype
        return [child1, child2]

    def crossover_ge(self, parent_one, parent_two):
        '''
        The crossover function for GE. This is the same function as it is for GA,
        However, determining the crossover point is different, as opposed to GA, 
        in GE the genotype is seperate to the phenotype.
        
        Here we determine a random gene in the genotype to be the 'slice' or 
        crossover point in which the genes between 0 and the crossover point
        from both parents will be recombinated and placed into two offspring.
        
        Parameters:
            parent_one (Neuroevolution obj) : The first parent chosen for reproduction.
            parent_two (Neuroevolution obj) : The second parent chosen for reproduction.
        
        Returns:
            list[Neuroevolution obj, Neuroevolution obj] : A list containing two Neuroevolution
                                                           objects. These will then be passed to
                                                           retrain.
        '''
        child1 = parent_one
        child2 = parent_two
        crossover_point = random.randint(0, len(list(child1.EA.genotype))-1)
        count = 0
        for i in range(len(child1.EA.genotype)-1):
            if count <= crossover_point:
                # Dont swap the fitness metrics
                temp_value = child1.EA.genotype[i]
                child1.EA.genotype[i] = child2.EA.genotype[i]
                child2.EA.genotype[i] = temp_value
            count += 1
        child1.EA.rectify_phenotype()
        child2.EA.rectify_phenotype()
        child1.phenotype = child1.EA.phenotype
        child2.phenotype = child2.EA.phenotype
        return [child1, child2]

    def mutate(self):
        '''
        Mutation operator that calls the objects EA.mutate() function for each objectin the population
        if a random value is less than or equal to the mutation rate set by the user.
        
        Once the genotypes of individuals are mutated, their neural architecture is rebuilt and
        they retrain.
        '''
        for individual in self.population:
            chance = random.randint(0, 100)
            if chance <= individual.mutation_rate:
                individual.EA.mutate()
                if len(individual.shape) > 2:
                    individual.model = individual.build_cnn_custom_architecture()
                    individual.run_cnn()
                else:
                    individual.model = individual.build_ann_custom_architecture()
                    individual.run_ann()
        return self

    def retrain_pop(self, population):
        '''
        Retrain function, used for a subset of population that have just been reproduced and haven't
        reran their neural network yet.
        
        These individuals are then added to a list to await transfer back to the population.

        Parameters:
            population list(Neuroevolution obj): A subset of the population that requires retraining.
        
        Returns:
            pop list(Neuroevolution obj): A subset of the population that has been retrained and ready to be put
                                          back into the population.
        '''
        pop = []
        for individual in population:
            if len(individual.shape) > 2:
                individual.model = individual.build_cnn_custom_architecture()
                individual.run_cnn()
            else:
                individual.model = individual.build_ann_custom_architecture()
                individual.run_ann()
            pop.append(individual)
        return pop

    def initialise_pop(self, insertions=[], elite_mode=False, grammar_file=None):
        '''
        This function initialises the population by creating NeuroEvolution objects and running them 
        for classification depending on the shape of the inputted data (ANN/CNN).
        
        Parameters:
            insertions (list(dict)) : Insertions allow for users to insert a phenotype if running with GA set,
                                      or input a genotype if running with GE set. These insertions allow the user
                                      to include a 'prior' best network should they have one.
            elite_mode (bool) : This variable determines whether the elite individual ie the fittest individual, should
                                automatically be cloned into the next generation.
                                
        '''
        console = Console()
        with console.status("[bold green]Initialising poulation...") as status:
            self.catch_eco()
            self.elite_mode=elite_mode
            if self.population_size:
                assert len(insertions) <= self.population_size, "Length of insertions must be smaller or equal to the population size"
            pop = []
            for genotype in insertions:
                a = Neuroevolution(evo_params=self.parameter_list, data=self.data, genotype=genotype, type=self.type, fittest=None,
                                    eco=self.eco, verbose=self.verbose, gene_value=self.gene_value, genotype_length=self.genotype_length,
                                    grammar_file=self.grammar_file)
                if len(a.shape) > 2:
                    a.run_cnn()
                else:
                    a.run_ann()
                pop.append(a)
            for _ in range(0, self.population_size-len(insertions)):
                a = Neuroevolution(evo_params=self.parameter_list, data=self.data, type=self.type, eco=self.eco, verbose=self.verbose, fittest=None,
                                gene_value=self.gene_value, genotype_length=self.genotype_length, grammar_file=self.grammar_file)
                if len(a.shape) > 2:
                    a.run_cnn()
                else:
                    a.run_ann()
                pop.append(a)
            
            self.population = pop
            console.log("Initialisation complete...")
        gc.collect()
        return self
    
    def which_fittest(self):
        '''
        This function discovers the fittest individual in the population, and saves the fittest individual as self.fittest.
        It also discovers the average fitness of the population and saves this as self.pop_average_fitness.
        '''
        fittest_val = 0.0
        self.pop_average_fitness = 0.0
        for individual in self.population:
            if self.fitness_function not in individual.EA.phenotype:
                print ('This individual doesn\'t have metrics... ', individual.EA.phenotype)
                if len(individual.shape) > 2:
                    individual.run_cnn()
                else:
                    individual.run_ann()

            if individual.EA.phenotype.get(self.fitness_function) >= fittest_val:
                self.fittest = individual
                fittest_val = individual.EA.phenotype.get(self.fitness_function)
                self.pop_average_fitness += fittest_val
            
        self.pop_average_fitness = self.pop_average_fitness / len(self.population) 
        self.catch_eco()
        return self

    def pop_recalibrate(self):
        '''
        This function recalibrates the population, specifically during ecological mode there are times within
        the evolutionary steps where the population size can be increased or decreased, this function ensures there are 
        the right amount of individuals within the population.
        
        If an ecological change happens and the population size is reduced, the least fit individuals will be removed from
        the population.
        '''
        self.catch_eco()
        while self.fittest.phenotype['population size'] > len(self.population):
            insertion = Neuroevolution(evo_params=self.parameter_list, data=self.data, type=self.type, eco=self.eco,
                                       fittest=self.fittest.phenotype, genotype=self.fittest.phenotype, verbose=self.verbose,
                                       gene_value=self.gene_value, genotype_length=self.genotype_length, grammar_file=self.grammar_file)
            if len(insertion.shape) > 2:
                insertion.run_cnn()
            else:
                insertion.run_ann()
            self.population.append(insertion)
        if self.fittest.EA.phenotype['population size'] < len(self.population):
            phenotype_list = []
            for individual in self.population:
                phenotype_list.append(individual.EA.phenotype)
            sorted_pop = sorted(phenotype_list, key= lambda x: x[self.fitness_function], reverse=True)
            while self.fittest.EA.phenotype['population size'] < len(sorted_pop):
                sorted_pop.pop()
            for phenotype in sorted_pop:
                insertion = Neuroevolution(evo_params=self.parameter_list, data=self.data, type=self.type, eco=self.eco, 
                                           fittest=self.fittest.phenotype, genotype=phenotype, verbose=self.verbose,
                                           gene_value=self.gene_value, genotype_length=self.genotype_length, grammar_file=self.grammar_file)
                if len(insertion.shape) > 2:
                    insertion.run_cnn()
                else:
                    insertion.run_ann()
                self.population.append(insertion)
        return self

    def output_results_into_csv(self, output_file, elite_individual):
        '''
        The function that outputs the elite individual into a csv file at program end into 
        a formatted output file.
        
        Parameters:
            output_file (str) : The user defined name they would like the directory of the output file to be.
                                    Files are stored as './Results/-output_file-'
            elite_individual (dic) : The phenotype of the best performing individual seen throughout
                                                        the evolutionary process so far.
        '''
        with open('./Results/'+output_file+'.csv','a') as fd:
            fd.write('FINAL OUTPUT' + "\n")
            fd.write('' + 'Hidden layers,' + 'Nodes,' + 'Activation functions,' + 'Optimiser,' +
                      'Epochs,' + 'B Size,' + 'Loss,' + 'Accuracy,' + 'F1,' + 'Precision,' + 'Recall,' + 'MAE,' + 'RMSE,' + 'Val. Acc.,' + 'Speed,' + 'Val x F1,' +  "\n") 
            fd.write(str(elite_individual.get('hidden layers')) + ',' + str(elite_individual.get('nodes')) + ',' + str(elite_individual.get('activation functions')) + ',' + 
                     str(elite_individual.get('optimiser')) + ',' + str(elite_individual.get('number of epochs')) + ',' +
                     str(elite_individual.get('batch size')) + ',' + str(elite_individual.get('loss')) + ',' + str(elite_individual.get('accuracy')) + ',' +
                     str(elite_individual.get('f1')) + ',' + str(elite_individual.get('precision')) + ',' + str(elite_individual.get('recall')) + ',' +
                     str(elite_individual.get('mae')) + ',' + str(elite_individual.get('rmse')) + ',' + str(elite_individual.get('validation_accuracy')) + ',' +
                     str(elite_individual.get('speed')) + ',' + str(elite_individual.get('val_acc_x_f1')) +"\n")
            fd.write("\n")
            fd.close()
        return None

    #This could be cleaned up.
    def checkpoint_handler(self, generation, elite_individual, output_file):
        '''
        A checkpoint function that currently outputs the elite individual to a csv file.
        This function is currently called every 50 generations, as it is currently useful to see improvements 
        over generations. But in future implementations it will be called every generation and will work
        in conjunction with self.initialise_pop() to insert the elite individual to the population.

        Parameters:
            generation (int) : The current evolutionary generation number.
            elite_individual (Dic) : The phenotype of the best performing individual seen throughout
                                                    the evolutionary process so far.
            output_file (str) : The user defined name they would like the directory of the output file to be.
                                Files are stored as './Results/-output_file-'
        '''
        string0 = "Validation accuracy = " + str(elite_individual['validation_accuracy'])
        string00 = "Speed = " + str(elite_individual['speed'])
        string1 = "MAE = " + str(elite_individual['mae'])
        string2 = "Test acc = " + str(elite_individual['accuracy'])
        string3 = "RMSE = " + str(elite_individual['rmse'])
        string4 = "Precision = " + str(elite_individual['precision'])
        try:
            string5 = "Recall = " + str(elite_individual['recall'])
        except ValueError:
            string5 = "ROC AUC Error"
        string6 = "F Measure score = " + str(elite_individual['f1'])
        string9 = "Validation accuracy x F Measure score = " + str(elite_individual['val_acc_x_f1'])
        string7 = "Individual = " + str(elite_individual)
        with open('./Results/'+output_file+'.csv','a') as fd:
            fd.write(generation + "\n") 
            fd.write(string0 + "\n")
            fd.write(string00 + "\n")
            fd.write(string1 + "\n")
            fd.write(string2 + "\n")
            fd.write(string3 + "\n")
            fd.write(string4 + "\n")
            fd.write(string5 + "\n")
            fd.write(string6 + "\n")
            fd.write(string7 + "\n")
            fd.write(string9 + "\n")
            fd.write("\n")
            fd.close()
        print ('Checkpoint stored')
        return self
    
    def plot(self, generation, best_fitness, elite_fitness, avg_fitness,
             output_file):
        '''
        Plotter function to plot the evolutions progress throughout generations.
        This function tracks: The Best fitness of the current generations population.
                              The elite individuals fitness, this is the individual 
                              that has performed best over the entire evolution.
                              The average fitness of the current generations population.

        Parameters:
            generation (int) : The current evolutionary generation number.
            best_fitness (float) : The best fitness of the current generations population.
            elite_fitness (float) : The fitness of the best performing individual seen throughout
                                    the evolutionary process so far.
            avg_fitness (float) : The average fitness of the current generations population.
        '''
        plt.plot(generation, best_fitness, label='Best fitness')
        plt.plot(generation, elite_fitness, label='Elite fitness')
        plt.plot(generation, avg_fitness, label='Average fitness')
        plt.legend(loc='best')
        plt.xlabel('Number of Generations')
        plt.ylabel('Fitness measured in '+str(self.fitness_function), size=12)
        filename = './Results/'+output_file
        i = 0
        while os.path.exists('{}{:d}.png'.format(filename, i)):
            i += 1
        plt.savefig('{}{:d}.png'.format(filename, i))
        plt.close()
        return self

    def load_data(self, data):
        '''
        Data loader function.
        
        Parameters:
            data (list(np.array)) : This should be a list of size 2 containing two numpy arrays,
                                    one containing training data and one comparing the training labels.
        '''
        dataX = data[0]
        dataY = data[1]
        self.data = dataX, dataY
        return self
    
    def set_fitness_function(self, fitness_function='f1'):
        '''
        Class setter function for self.fitness_function

        Parameters:
            fitness_function (str) Default ('f1') : User defined metric they want to be the networks fitness function.
        '''
        self.fitness_function = fitness_function
        return self

    def selection_choice(self):
        '''
        Checks user input (self.selection) for the selection operator and runs said functions.
        '''
        if self.selection.lower() == 'tournament':
            self.tournament_selection()
        elif self.selection.lower() == 'roulette':
            self.roulette_selection()
        return self

    def run(self, plot=True, verbose=0):
        '''
        Starter pistol function to run the evolutionary process. 
        Runs through the entire evolutionary process, selection, mutation, crossover etc.
        This function also calls the plot function to show the progression of the fittest
        individuals over generations.
        
        Parameters:
            plot (bool) : A boolean parameter set by the user to plot the results of the evolutionary process.
            verbose (int) (0/1/2) : A flag parameter used to turn tensorflows verbose function on.
                                    See here for more help. https://www.tensorflow.org/api_docs/python/tf/keras/Model
        '''
        self.verbose=verbose
        console = Console()
        catch = False
        self.which_fittest()
        self.catch_eco()
        
        assert self.max_generations > 0, 'Maximum number of generations must be > 0'
        assert verbose in [0,1,2], 'Verbose must be 0, 1 or 2. See here for more help. https://www.tensorflow.org/api_docs/python/tf/keras/Model'
        plot_generation, plot_best_fitness, plot_elite_fitness, plot_elite_fitness, plot_avg_fitness = ([] for i in range(5))
        elite_individual = None
        elite_fitness = 0.0
        output_file = self.dataset_name+'_'+str(self.type)+'_p_'+str(self.population_size)+'_mr_'+str(self.mutation_rate)+'_cr_'+str(self.cloning_rate)+'_eco_'+str(self.eco)
        with console.status("[bold green]Running through generations...") as status:
            i = 1
            try:
                while i <= self.max_generations:
                    self.selection_choice()
                    self.mutate()
                    if self.eco:
                        self.catch_eco()
                        self.pop_recalibrate()
                        if self.max_generations <= i:
                            catch = True
                    self.which_fittest()
                    if self.fittest.EA.phenotype.get(self.fitness_function) > elite_fitness:
                        print ('Elite individual = ', elite_individual)
                        print ('Elite fitness = ', elite_fitness)
                        elite_individual = self.fittest.EA.phenotype
                        elite_fitness = elite_individual[self.fitness_function]
                        print ('Elite individual = ', elite_individual)
                        print ('Elite fitness = ', elite_fitness)
                    #Every 50th generation, save the fittest network in a file.
                    if i % 50 == 0 or i == 1 or catch:
                        print ('Elite individual before checkpoint = ', elite_individual)
                        print ('Elite fitness before checkpoint = ', elite_fitness)
                        self.checkpoint_handler(str(i), elite_individual=elite_individual, output_file=output_file)
                    plot_generation.append(i)
                    plot_best_fitness.append(self.fittest.EA.phenotype[self.fitness_function])  
                    plot_elite_fitness.append(elite_fitness)  
                    plot_avg_fitness.append(self.pop_average_fitness)  

                    console.log(f"Generation {i} complete...")
                    if catch == True: 
                        break
                    i += 1   
                if plot:
                    self.plot(generation=plot_generation, best_fitness=plot_best_fitness, elite_fitness=plot_elite_fitness, 
                            avg_fitness=plot_avg_fitness, output_file=output_file)
                global highest
                highest = 0
            except KeyboardInterrupt:
                self.checkpoint_handler(str(i), elite_individual=elite_individual, output_file=output_file)
                self.plot(generation=plot_generation, best_fitness=plot_best_fitness, elite_fitness=plot_elite_fitness, 
                            avg_fitness=plot_avg_fitness, output_file=output_file)
            self.output_results_into_csv(output_file, elite_individual)
        completion_message = '***Evolution complete***'
        print (completion_message)
        gc.collect()
        return self
