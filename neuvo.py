import time
import math
import random
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
import json
from rich.console import Console
import copy
import numpy as np
import gc
from GA import GA
import warnings
import logging
from datetime import datetime
warnings.filterwarnings('ignore') 
tf.get_logger().setLevel(logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from Metrics import f1_m, precision_m, recall_m
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
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
        '''
        A function to check the users data shape.
        If the shapes dimensions are greater than 2 then a CNN will be built,
        else an artificial neural network will be built.

        '''
        if len(self.shape) > 2:
            self.build_cnn_custom_architecture()
        else:
            self.build_ann_custom_architecture()
        return self
    
    def build_ea(self, genotype_length=32, gene_value=40):
        '''
        A function to check the users preference for evolutionary algorithms.

        Parameters:
            genotype_length (int) : The length of the genotype which will map to a phenotype when running
                                    alongside grammatical evolution.
            gene_value (int) : The maximum integer value for each gene in the genotype.
        '''
        if self.type.lower() == 'ga':
            self.EA = GA(shape=self.shape, mutation_rate=self.mutation_rate, phenotype=self.genotype, eco=self.eco)
        else:
            self.EA = GE(shape=self.shape, mutation_rate=self.mutation_rate, genotype=self.genotype,
                        genotype_length=genotype_length, gene_value=gene_value, user_grammar_file=self.grammar_file)

    def build_parent(self, genotype_length=32, gene_value=40):
        '''
        A function to initialise a parent individual when the object is first created.

        Parameters:
            genotype_length (int) : The length of the genotype which will map to a phenotype when running
                                    alongside grammatical evolution.
            gene_value (int) : The maximum integer value for each gene in the genotype.
        '''
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
            self.mutation_rate = self.fittest.EA.phenotype['mutation rate']
            self.population_size = self.fittest.EA.phenotype['population size']
            self.cloning_rate = self.fittest.EA.phenotype['cloning rate']
            self.max_generations = self.fittest.EA.phenotype['max generations']
        else:
            self.mutation_rate = self.parameter_list['mutation rate']
            self.population_size = self.parameter_list['population size']
            self.cloning_rate = self.parameter_list['cloning rate']
            self.max_generations = self.parameter_list['max generations']
        return self
    
    def custom(self, tensor):
        '''
        A function to create custom activation functions using Pythons `eval' functionality.

        Parameters:
            tensor (string) : A string representation of an activation function e.g. 'min(x,0)'
        Returns:
            model (sequential model) : A Tensorflow sequential model built for classifying 2D data.
        '''
        from tensorflow import nn
        sub_string = self.string
        try:
            x = eval(sub_string)
        except NameError:
            x = eval("nn."+sub_string+"(tensor)")
        return x
    
    def build_ann_custom_architecture(self):
        '''
        A function to build an artificial neural network using Tensorflows Sequential model,
        with the objects architecture parameters. This model gets passed to another function to run the model
        on the users data.

        Returns:
            model (sequential model) : A Tensorflow sequential model built for classifying 2D data.
        '''
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
    
    def run_ann(self, L=None, data=None):
        '''
        A function to run a convolutional neural network using Tensorflows Sequential model.
        Uses 2 fold cross validation and the objects architecture parameters.

        Parameters:
            L (multiprocessing list) A list to append the NeuroEvolution object used for multi training.
            data (numpy array) Numpy array consisting of the data and labels.
        '''
        dataX = data[0]
        dataY = data[1]
        tf.keras.backend.clear_session()
        num_folds = 5
        model = self.build_ann_custom_architecture()
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3)
        start = time.time()
        loss, accuracy, f1, precision, recall, mae, rmse = (0.0,)*7
        for i, (train_index, test_index) in enumerate(kfold.split(dataX, dataY)):
            X_train,X_test = dataX[train_index],dataX[test_index]
            Y_train,Y_test = dataY[train_index],dataY[test_index]
            model.fit(X_train, Y_train, batch_size=self.EA.phenotype['batch size'], epochs=self.EA.phenotype['number of epochs'],
                            verbose=self.verbose, validation_data=(X_test, Y_test), callbacks=[es, TerminateOnNaN()])
            history = model.history.history
            last_val = history['val_accuracy'].pop()
            los, acc, f, prec, rec, ma, rms = model.evaluate(X_test, Y_test, verbose=self.verbose)
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
        if L != None:
            L.put(self)
        return self
    
    def build_cnn_custom_architecture(self):
        '''
        A function to build a convolutional neural network using Tensorflows Sequential model,
        with the objects architecture parameters. This model gets passed to another function to run the model
        on the users data.

        Returns:
            model (sequential model) : A Tensorflow sequential model built for classifying 2D data.
        '''
        tf.keras.backend.clear_session()
        model = Sequential()
        try:
            model.add(Conv2D(self.EA.phenotype['nodes']/2, kernel_size=(3, 3), padding='same', activation=self.EA.phenotype['activation functions'][0], input_shape=self.dataX.shape[1:]))
            model.add(BatchNormalization())
            model.add(Conv2D(self.EA.phenotype['nodes']/2, kernel_size=(3, 3), padding='same', activation=self.EA.phenotype['activation functions'][0]))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for i in range(1, self.EA.phenotype['hidden layers']):
                self.string = self.EA.phenotype['activation functions'][i]
                model.add(Conv2D(self.EA.phenotype['nodes'], (3, 3), padding='same', activation=self.EA.phenotype['activation functions'][i]))
                model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(self.EA.phenotype['nodes']*2, kernel_size=(3, 3), padding='same', activation=self.EA.phenotype['activation functions'][-2]))
            model.add(BatchNormalization())
            model.add(Conv2D(self.EA.phenotype['nodes']*2, kernel_size=(3, 3), padding='same', activation=self.EA.phenotype['activation functions'][-2]))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dropout(0.2))

            model.add(Dense(1024, activation=self.EA.phenotype['activation functions'][-2]))
            model.add(Dropout(0.2))

            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer=str(self.EA.phenotype['optimiser']), loss='binary_crossentropy',
                       metrics=['accuracy', Precision(), Recall(), MeanAbsoluteError(), RootMeanSquaredError()])
        except ValueError:
            self.string = self.EA.phenotype['activation functions'][0]
            get_custom_objects().update({'custom': self.custom})
            model.add(Conv2D(self.EA.phenotype['nodes']/2, kernel_size=(3, 3), padding='same', activation=self.custom, input_shape=self.dataX.shape[1:]))
            model.add(BatchNormalization())
            model.add(Conv2D(self.EA.phenotype['nodes']/2, kernel_size=(3, 3), padding='same', activation=self.custom))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for i in range(1, self.EA.phenotype['hidden layers']):
                self.string = self.EA.phenotype['activation functions'][i]
                model.add(Conv2D(self.EA.phenotype['nodes'], (3, 3), padding='same', activation=self.custom))
                model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))

            self.string = self.EA.phenotype['activation functions'][-2]
            model.add(Conv2D(self.EA.phenotype['nodes']*2, kernel_size=(3, 3), padding='same', activation=self.custom))
            model.add(BatchNormalization())
            model.add(Conv2D(self.EA.phenotype['nodes']*2, kernel_size=(3, 3), padding='same', activation=self.custom))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dropout(0.2))

            model.add(Dense(1024, activation=self.custom))
            model.add(Dropout(0.2))

            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer=str(self.EA.phenotype['optimiser']), loss='binary_crossentropy',
                       metrics=['accuracy', Precision(), Recall(), MeanAbsoluteError(), RootMeanSquaredError()])
        return model

    def run_cnn(self, L=None, data=None):
        '''
        A function to run a convolutional neural network using Tensorflows Sequential model.
        Uses 5 fold cross validation and the objects architecture parameters. Metrics are stored in the individuals
        phenotype.

        Parameters:
            L (multiprocessing list) : A list to append the NeuroEvolution object used for multi training.
            data (numpy array) : Numpy array consisting of the data and labels.
        '''
        dataX = data[0]
        dataY = data[1]
        tf.keras.backend.clear_session()
        num_folds = 5
        model = self.build_cnn_custom_architecture()
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=2)
        start = time.time()
        loss, accuracy, f1, precision, recall, mae, rmse = (0.0,)*7
        for i, (train_index, test_index) in enumerate(kfold.split(dataX, dataY)):
            X_train,X_test = dataX[train_index],dataX[test_index]
            Y_train,Y_test = dataY[train_index],dataY[test_index]
            model.fit(X_train, Y_train, batch_size=self.EA.phenotype['batch size'], epochs=self.EA.phenotype['number of epochs'], 
                           verbose=self.verbose, validation_data=(X_test, Y_test), callbacks=[es, TerminateOnNaN()])
            history = model.history.history
            last_val = history['val_accuracy'].pop()
            los, acc, prec, rec, ma, rms = model.evaluate(X_test, Y_test, verbose=self.verbose)
            loss += los
            accuracy += acc
            f1 += 2*(rec*prec)/1+(rec+prec)
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
        if L != None:
            L.append(self)
        return None

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
        #self.catch_eco()
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
        if self.eco == True and self.type.lower() == 'ge':
            possible = False
        else:
            possible = True
        assert possible == True, 'Sorry, '+str(self.type)+' and eco mode has not been implemented yet.'
        return self

    def catch_eco(self):
        '''
        A function to check whether eco mode is activated, if it is then the evolutionary parameters will be
        set by the fittest individuals genes, else it will be chosen by the user from a parameter_file or 
        inputted at runtime by the user.
        '''

        if self.eco and self.fittest != None:
            self.mutation_rate = self.fittest.EA.phenotype['mutation rate']
            self.cloning_rate = self.fittest.EA.phenotype['cloning rate']
            self.population_size = self.fittest.EA.phenotype['population size']
            self.max_generations = self.fittest.EA.phenotype['max generations']

            self.parameter_list = {'mutation rate' : self.mutation_rate, 
                                   'population size' : self.population_size, 
                                   'cloning rate' : self.cloning_rate,
                                   'max generations' : self.max_generations}
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

    def tournament_selection(self):
        '''
        The tournament selection operator, where n random individuals are chosen 
        from the population, the two fittest individuals are then eligible for reproduction.
        
        Parameters:
            tournament_size (int) : The number of individuals in the tournament.
        
        Returns:
            NeuvoBuilder obj : A NeuvoBuilder object.
        '''
        
        retrain_pop = []
        population_copy = copy.copy(self.population)
        cloned_pop = []
        if self.elite_mode:
            cloned_pop.append(copy.deepcopy(self.fittest))
        
        n = round((len(population_copy)*self.cloning_rate))-len(cloned_pop)
        cloned_pop.extend(population_copy[:n])
        reproducible_pop = population_copy[n:]
        if self.eco:
            self.tournament_size = math.ceil(len(reproducible_pop) / 2) + 1
        
        assert self.tournament_size <= len(reproducible_pop), "Tournament size must be less than or equal to the size of the population." + str(len(reproducible_pop))
        j = len(retrain_pop)
        while j < self.population_size-len(cloned_pop):
            sample = random.sample(reproducible_pop, self.tournament_size)
            random_choices = []
            sample.sort(key =lambda x: x.EA.phenotype.get(self.fitness_function), reverse=True)
            random_choices.append(sample[0])
            random_choices.append(sample[1])
            child1, child2 = self.crossover(random_choices[0], random_choices[1])
            retrain_pop.append(child1)
            j += 1
            retrain_pop.append(child2)
            j += 1
        new_pop = []
        temp_pop = retrain_pop
        phenotype_list = []
        to_be_cloned_pop = []
        for individual in temp_pop:
            phenotype = individual.EA.phenotype
            if phenotype in phenotype_list:
                temp_pop.remove(individual)
            else:
                phenotype_list.append(phenotype)
        
        temp_pop = self.retrain_pop(temp_pop, data=self.data)
        phenotype_list = []
        for individual in temp_pop:
            phenotype = individual.EA.phenotype
            if list(phenotype)[-10:] in phenotype_list:
                to_be_cloned_pop.append(individual)
                temp_pop.remove(individual)
            else:
                phenotype_list.append(list(phenotype)[-10:])
        new_pop.extend(cloned_pop)
        new_pop.extend(to_be_cloned_pop)
        new_pop.extend(temp_pop)
        self.population = new_pop
        self.catch_eco()
        return self

    def roulette_selection(self):
        '''
        The Roulette wheel selection operator, this function allows each genotype to be selected,
        but offers preferential treatment to fitter genotypes.
        
        Returns:
            NeuvoBuilder obj : A NeuvoBuilder object.
        '''
        cloned_pop = []
        if self.elite_mode:
            cloned_pop.append(copy.copy(self.fittest))
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
        p1 = copy.deepcopy(parent_one)
        p2 = copy.deepcopy(parent_two)
        p1.EA.remove_metrics()
        p2.EA.remove_metrics()
        if self.type == 'ge':
            children = self.crossover_ge(parent_one=p1, parent_two=p2)
        elif self.type == 'ga':
            children = self.crossover_ga(parent_one=p1, parent_two=p2)
        return children
    
    def crossover_ga(self, parent_one, parent_two):
        '''
        The crossover function for GA. This is the same function as it is for GE,
        However, determining the crossover point is different, as opposed to GE, 
        in GA the genotype is the same as the phenotype.
        
        Parameters:
            parent_one (Neuroevolution obj) : The first parent chosen for reproduction.
            parent_two (Neuroevolution obj) : The second parent chosen for reproduction.
        
        Returns:
            list[Neuroevolution obj, Neuroevolution obj] : A list containing two Neuroevolution
                                                           objects. These will then be passed to
                                                           retrain.
        '''
        p1 = parent_one
        p2 = parent_two
        if self.crossover_method == 'one_point':
            child1, child2 = self.one_point_crossover_ga(parent_one=p1, parent_two=p2)
        elif self.crossover_method == 'two_point':
            child1, child2 = self.two_point_crossover_ga(parent_one=p1, parent_two=p2)
        return [child1, child2]

    def crossover_ge(self, parent_one, parent_two):
        '''
        The crossover function for GE. This is the same function as it is for GA,
        However, determining the crossover point is different, as opposed to GA, 
        in GE the genotype is seperate to the phenotype.
        
        Parameters:
            parent_one (Neuroevolution obj) : The first parent chosen for reproduction.
            parent_two (Neuroevolution obj) : The second parent chosen for reproduction.
        
        Returns:
            list[Neuroevolution obj, Neuroevolution obj] : A list containing two Neuroevolution
                                                           objects. These will then be passed to
                                                           retrain.
        '''
        p1 = parent_one
        p2 = parent_two
        if self.crossover_method == 'one_point':
            child1, child2 = self.one_point_crossover_ge(parent_one=p1, parent_two=p2)
        elif self.crossover_method == 'two_point':
            child1, child2 = self.two_point_crossover_ge(parent_one=p1, parent_two=p2)
        return [child1, child2]

    def one_point_crossover_ga(self, parent_one, parent_two):
        '''
        Here we determine a random gene in the genotype to be the 'slice' or 
        crossover point in which the genes between 0 and the crossover point
        from both parents will be recombinated and placed into two offspring.
        
        Parameters:
            parent_one (Neuroevolution obj) : The first parent chosen for reproduction.
            parent_two (Neuroevolution obj) : The second parent chosen for reproduction.
        
        Returns:
            tuple(Neuroevolution obj, Neuroevolution obj) : A tuple containing two Neuroevolution
                                                           objects. These will then be passed to
                                                           retrain.
        '''
        child1 = parent_one
        child2 = parent_two
        if len(list(child1.EA.phenotype.items())) <= len(list(child2.EA.phenotype.items())):
            smallest = child1
            largest = child2
        else:
            smallest = child2
            largest = child1
        crossover_point = random.randint(0, len(list(smallest.EA.phenotype.items()))-2)
        count = 0
        for key in smallest.EA.phenotype:
            if count <= crossover_point:
                temp_value = smallest.EA.phenotype[key]
                smallest.EA.phenotype[key] = largest.EA.phenotype[key]
                largest.EA.phenotype[key] = temp_value
            count += 1
        child1 = smallest
        child2 = largest
        child1.EA.rectify_phenotype()
        child2.EA.rectify_phenotype()
        child1.phenotype = child1.EA.phenotype
        child2.phenotype = child2.EA.phenotype
        return child1, child2

    def two_point_crossover_ga(self, parent_one, parent_two):
        '''
        The two point crossover function for GA. 
        
        Here we determine two random points in the genotype to be the crossover points and
        the genes between these two are placed into the offspring.
        
        Parameters:
            parent_one (Neuroevolution obj) : The first parent chosen for reproduction.
            parent_two (Neuroevolution obj) : The second parent chosen for reproduction.
        
        Returns:
            tuple(Neuroevolution obj, Neuroevolution obj) : A tuple containing two Neuroevolution
                                                           objects. These will then be passed to
                                                           retrain.
        '''
        child1 = parent_one
        child2 = parent_two
        if len(list(child1.EA.phenotype.items())) <= len(list(child2.EA.phenotype.items())):
            smallest = child1
            largest = child2
        else:
            smallest = child2
            largest = child1
        first_crossover_point = random.randint(0, len(list(smallest.EA.phenotype.items()))/2)
        second_crossover_point = random.randint(first_crossover_point, len(list(smallest.EA.phenotype.items()))-1)
        count = first_crossover_point
        for key in smallest.EA.phenotype:
            if count <= second_crossover_point:
                temp_value = smallest.EA.phenotype[key]
                smallest.EA.phenotype[key] = largest.EA.phenotype[key]
                largest.EA.phenotype[key] = temp_value
            count += 1
        child1 = smallest
        child2 = largest
        child1.EA.rectify_phenotype()
        child2.EA.rectify_phenotype()
        child1.phenotype = child1.EA.phenotype
        child2.phenotype = child2.EA.phenotype
        return child1, child2
    
    def one_point_crossover_ge(self, parent_one, parent_two):
        '''
        The one point crossover function for GE. This is the same function as it is for GA,
        However, determining the crossover point is different, as opposed to GA, 
        in GE the genotype is seperate to the phenotype.
        
        Here we determine a random gene in the genotype to be the 'slice' or 
        crossover point in which the genes between 0 and the crossover point
        from both parents will be recombinated and placed into two offspring.
        
        Parameters:
            parent_one (Neuroevolution obj) : The first parent chosen for reproduction.
            parent_two (Neuroevolution obj) : The second parent chosen for reproduction.
        
        Returns:
            tuple(Neuroevolution obj, Neuroevolution obj) : A tuple containing two Neuroevolution
                                                           objects. These will then be passed to
                                                           retrain.
        '''
        child1 = parent_one
        child2 = parent_two
        crossover_point = random.randint(0, len(list(child1.EA.genotype))-2)
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
        return child1, child2

    def two_point_crossover_ge(self, parent_one, parent_two):
        '''
        The two point crossover function for GE. This is the same function as it is for GA,
        However, determining the crossover point is different, as opposed to GA, 
        in GE the genotype is seperate to the phenotype.
        
        Here we determine two random points in the genotype to be the crossover points and
        the genes between these two are placed into the offspring.
        
        Parameters:
            parent_one (Neuroevolution obj) : The first parent chosen for reproduction.
            parent_two (Neuroevolution obj) : The second parent chosen for reproduction.
        
        Returns:
            tuple(Neuroevolution obj, Neuroevolution obj) : A tuple containing two Neuroevolution
                                                           objects. These will then be passed to
                                                           retrain.
        '''
        child1 = parent_one
        child2 = parent_two
        first_crossover_point = random.randint(0, len(list(child1.EA.genotype))-2)
        second_crossover_point = random.randint(first_crossover_point, len(list(child1.EA.genotype))-1)
        count = first_crossover_point
        for i in range(len(child1.EA.genotype)-1):
            if count <= second_crossover_point:
                # Dont swap the fitness metrics
                temp_value = child1.EA.genotype[i]
                child1.EA.genotype[i] = child2.EA.genotype[i]
                child2.EA.genotype[i] = temp_value
            count += 1
        child1.EA.rectify_phenotype()
        child2.EA.rectify_phenotype()
        child1.phenotype = child1.EA.phenotype
        child2.phenotype = child2.EA.phenotype
        return child1, child2

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
                    individual.run_cnn(data=self.data)
                else:
                    individual.run_ann(data=self.data)
        return self

    def multi_train_cnn(self, population, data):
        '''
        Retrain function, used for a subset of population that have just been reproduced and haven't
        reran their convolutional neural network yet.
        
        These individuals are then added to a list to await transfer back to the population.

        Parameters:
            population list(Neuroevolution obj): A subset of the population that requires retraining.
            data (numpy array) : The users data.
        
        Returns:
            population list(Neuroevolution obj): A subset of the population that has been retrained and ready to be put
                                          back into the population.
        '''
        pop = []
        j = 0
        left = len(population)
        cpu_count = multiprocessing.cpu_count()
        for individual in population:
            individual.run_cnn(L=None, data=data)
        # with multiprocessing.Manager() as manager:
        #     L = manager.list()
        #     processes = []
        #     while j < len(population):
        #         if left < int(cpu_count):
        #             cpu_count = left
        #         for i in range(cpu_count):
        #             p = multiprocessing.Process(target=population[j].run_cnn, args=(L, ))
        #             p.start()
        #             processes.append(p)
        #             j += 1
        #             left -= 1
        #     for process in processes:
        #         process.join()
        #     pop = list(L)
        return population

    def multi_train_ann(self, population, data):
        '''
        Retrain function, used for a subset of population that have just been reproduced and haven't
        reran their neural network yet.
        
        These individuals are then added to a list to await transfer back to the population.

        Parameters:
            population list(Neuroevolution obj): A subset of the population that requires retraining.
            data (numpy array) : The users data.
        
        Returns:
            population list(Neuroevolution obj): A subset of the population that has been retrained and ready to be put
                                          back into the population.
        '''
        pop = []
        j = 0
        left = len(population)
        cpu_count = multiprocessing.cpu_count()
        for individual in population:
            individual.run_ann(L=None, data=data)

        # with multiprocessing.Manager() as manager:
        #     L = manager.list()
        #     processes = []
        #     queue1 = multiprocessing.Queue()
        #     while j < len(population):
        #         if left < int(cpu_count):
        #             cpu_count = left
        #         for i in range(cpu_count):
        #             p = multiprocessing.Process(target=population[j].run_ann, args=(queue1, data))
        #             p.start()
        #             print ('before get...')
        #             pop.append(queue1.get())
        #             print ('after get...')
        #             processes.append(p)
        #             j += 1
        #             left -= 1
        #     for process in processes:
        #         process.join()
            #pop = list(L)
        return population

    def retrain_pop(self, population, data):
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
        if population:
            if len(population[0].shape) > 2:
                self.population = self.multi_train_cnn(population, data)
            else:
                self.population = self.multi_train_ann(population, data)
        return self.population

    def initialise_pop(self, insertions=[], elite_mode=False, grammar_file=None):
        '''
        This function initialises the population by creating NeuroEvolution objects and running them 
        for classification depending on the shape of the inputted data (ANN/CNN).
        
        Parameters:
            insertions (neuro_objects) : Insertions allow for users to insert a genotype/phenotype. These insertions allow the user
                                      to restart their run or include a 'prior' best network should they have one.
            elite_mode (bool) : This variable determines whether the elite individual ie the fittest individual, should
                                automatically be cloned into the next generation.
                                
        '''
        console = Console()
        with console.status("[bold green]Initialising population...") as status:
            self.catch_eco()
            self.elite_mode=elite_mode
            if self.population_size:
                assert len(insertions) <= self.population_size, "Length of insertions must be smaller or equal to the population size"
            pop = []
            for phenotype in insertions:
                a = Neuroevolution(evo_params=self.parameter_list, data=self.data, phenotype=phenotype, type=self.type, fittest=None,
                                    eco=self.eco, verbose=self.verbose, gene_value=self.gene_value, genotype_length=self.genotype_length,
                                    grammar_file=self.grammar_file)
                pop.append(a)
            for _ in range(0, self.population_size-len(insertions)):
                a = Neuroevolution(evo_params=self.parameter_list, data=self.data, type=self.type, eco=self.eco, verbose=self.verbose, fittest=None,
                                gene_value=self.gene_value, genotype_length=self.genotype_length, grammar_file=self.grammar_file)
                pop.append(a)
            if len(pop[0].shape) > 2:
                self.population = self.multi_train_cnn(pop)
            else:
                self.population = self.multi_train_ann(pop, data=self.data)
            console.log("Initialisation complete...")
        self.catch_eco()
        gc.collect()
        return self
    
    def which_fittest(self):
        '''
        This function discovers the fittest individual in the population, and saves the fittest individual as self.fittest.
        It also discovers the average fitness of the population and saves this as self.pop_average_fitness.
        '''
        fittest_val = -1.0
        fittest_of_gen = None
        self.pop_average_fitness = 0.0
        
        for individual in self.population:
            if self.fittest is None:
                self.fittest = individual
            if individual.EA.phenotype.get(self.fitness_function) > fittest_val:
                fittest_of_gen = individual
                fittest_val = individual.EA.phenotype.get(self.fitness_function)
                self.pop_average_fitness += fittest_val
                if fittest_of_gen.EA.phenotype.get(self.fitness_function) > self.fittest.EA.phenotype.get(self.fitness_function):
                    self.fittest = fittest_of_gen
        self.pop_average_fitness = self.pop_average_fitness / len(self.population) 
        self.catch_eco()
        return fittest_of_gen
    
    def save_phenotypes(self):
        '''
        Checkpoint function to save the populations phenotypes into a .txt file.
        '''
        pop_to_save = []
        for individual in self.population:
            if self.type == 'ga':
                pop_to_save.append(individual.EA.phenotype)
            elif self.type == 'ge':
                pop_to_save.append(individual.EA.genotype)
        with open('individuals_saved.txt', 'a') as output_file:
            output_file.write(self.dataset_name+' '+str(datetime.now()))
            output_file.write(json.dumps(pop_to_save))
            output_file.write('\n')
        return self

    def pop_recalibrate(self):
        '''
        This function recalibrates the population, specifically during ecological mode there are times within
        the evolutionary steps where the population size can be increased or decreased, this function ensures there are 
        the right amount of individuals within the population.
        
        If an ecological change happens and the population size is reduced, the least fit individuals will be removed from
        the population.
        '''
        print ('pop size before = ', len(self.population))
        self.which_fittest()
        pop = []
        if self.fittest.EA.phenotype['population size'] > len(self.population):
            difference = self.fittest.EA.phenotype['population size'] - len(self.population)
            for _ in range(difference):
                print ('Adding a new individual!!!')
                insertion = Neuroevolution(evo_params=self.parameter_list, data=self.data, type=self.type, eco=self.eco,
                                        fittest=self.fittest, verbose=self.verbose,
                                        gene_value=self.gene_value, genotype_length=self.genotype_length, grammar_file=self.grammar_file)
                pop.append(insertion)
        if self.fittest.EA.phenotype['population size'] < len(self.population):
            phenotype_list = []
            for individual in self.population:
                phenotype_list.append(individual.EA.phenotype)
            sorted_pop = sorted(phenotype_list, key= lambda x: x[self.fitness_function], reverse=False)
            print ('Sorted pop order = ', sorted_pop)
            print ('Poppping individuals off')
            for i, o in enumerate(self.population):
                if self.fittest.EA.phenotype['population size'] < len(self.population):
                    if o.EA.phenotype == sorted_pop[i]:
                        print ('i = ', i)
                        print ('o = ', o)
                        del self.population[i]
                else:
                    break
            # for phenotype in sorted_pop:
            #     insertion = Neuroevolution(evo_params=self.parameter_list, data=self.data, type=self.type, eco=self.eco, 
            #                                fittest=self.fittest, genotype=phenotype, verbose=self.verbose,
            #                                gene_value=self.gene_value, genotype_length=self.genotype_length, grammar_file=self.grammar_file)
            #     pop.append(insertion)
        train_pop = pop
        if train_pop: 
            if len(train_pop[0].shape) > 2:
                trained_pop = self.multi_train_cnn(train_pop, data=self.data)
                self.population.extend(trained_pop)
            else:
                print ('Sending to multi train...')
                trained_pop = self.multi_train_ann(train_pop, data=self.data)
                self.population.extend(trained_pop)
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
            fd.write('Elite individual' + "\n")
            fd.write(str(elite_individual))
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
        self.save_phenotypes()
        string7 = "Individual = " + str(elite_individual)
        pop_avg_fitness = str(self.pop_average_fitness)
        with open('./Results/'+output_file+'.csv','a') as fd:
            fd.write(generation + "\n") 
            fd.write(string7 + "\n")
            fd.write('Average population fitness: ' + pop_avg_fitness + "\n")
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
            data (list(np.array, np.array)) : This should be a list of size 2 containing two numpy arrays,
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

    def run(self, plot=True):
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
        #self.verbose=verbose
        console = Console()
        catch = False
        self.which_fittest()
        self.catch_eco()
        
        assert self.max_generations > 0, 'Maximum number of generations must be > 0'
        assert self.verbose in [0,1,2], 'Verbose must be 0, 1 or 2. See here for more help. https://www.tensorflow.org/api_docs/python/tf/keras/Model'
        plot_generation, plot_best_fitness, plot_elite_fitness, plot_elite_fitness, plot_avg_fitness = ([] for i in range(5))
        output_file = self.dataset_name+'_'+str(self.type)+'_p_'+str(self.population_size)+'_mr_'+str(self.mutation_rate)+'_cr_'+str(self.cloning_rate)+'_eco_'+str(self.eco)
        with console.status("[bold green]Running through generations...") as status:
            i = 1
            try:
                while i <= self.max_generations:
                    self.selection_choice()
                    self.mutate()
                    if self.eco:
                        self.pop_recalibrate()
                        if self.max_generations <= i:
                            catch = True
                    fittest_of_gen = self.which_fittest()

                    # for individual in self.population:
                    #     print (individual.EA.phenotype.get(self.fitness_function))
                    # print ('Fittest individual = ', self.fittest.EA.phenotype)
                    # print ('Current parameters = ', self.parameter_list)
                    # print ('Current parameters outside = ', {'max_generations' : self.max_generations, 'cloning_rate' : self.cloning_rate})

                    #Every 50th generation, save the fittest network in a file.
                    if i % 5 == 0 or i == 1 or catch:
                        self.checkpoint_handler(str(i), elite_individual=self.fittest.EA.phenotype, output_file=output_file)
                    plot_generation.append(i)
                    plot_best_fitness.append(fittest_of_gen.EA.phenotype.get(self.fitness_function))  
                    plot_elite_fitness.append(self.fittest.EA.phenotype.get(self.fitness_function))  
                    plot_avg_fitness.append(self.pop_average_fitness)  
                    console.log(f"Generation {i} complete...")
                    if catch == True: 
                        break
                    gc.collect()
                    i += 1   
                if plot:
                    self.plot(generation=plot_generation, best_fitness=plot_best_fitness, elite_fitness=plot_elite_fitness, 
                            avg_fitness=plot_avg_fitness, output_file=output_file)
                global highest
                highest = 0
            except KeyboardInterrupt:
                self.checkpoint_handler(str(i), elite_individual=self.fittest.EA.phenotype, output_file=output_file)
                self.plot(generation=plot_generation, best_fitness=plot_best_fitness, elite_fitness=plot_elite_fitness, 
                            avg_fitness=plot_avg_fitness, output_file=output_file)
            self.output_results_into_csv(output_file, self.fittest.EA.phenotype)
        completion_message = '***Evolution complete***'
        gc.collect()
        return self