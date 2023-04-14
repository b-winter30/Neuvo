import collections
import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import random
import multiprocessing
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
import warnings
import logging
warnings.filterwarnings('ignore') 
tf.get_logger().setLevel(logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import Metrics
import math
import pandas as pd
import math
from Metrics import  precision_m, recall_m, f1_m
import re

highest = 0
best_in_pop = []
#population = {}
activation_functions = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu',
                        'elu']
optimizers = ['SGD', 'RMSprop', 'Adam',
              'Adadelta', 'Adagrad', 'Adamax', 'Nadam']
epochs = [50, 100, 150, 200, 250]
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def open_dataset(directory):
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    dataX = pd.read_csv(directory+'/x_data.csv', header=None)
    dataY = pd.read_csv(directory+'/y_data.csv', header=None)
    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, shuffle=True)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=True) 
    return (X_train, Y_train, X_test, Y_test, X_val, Y_val)

def createAndReturnFitness(noOfHiddenLayers, numberOfHiddenLayerNodes, af, optimizer, epochs, batch_size, data, que):
    import psutil
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from Metrics import  precision_m, recall_m, f1_m
    # Starting the network
    tf.keras.backend.clear_session()
    #X_train, Y_train, X_test, Y_test, X_val, Y_val = data
    dataX, dataY = data
    dataX = dataX.to_numpy()
    dataY = dataY.to_numpy()
    start = time.time()
    af1 = af[0]
    af2 = af[1]
    af3 = af[2]
    opti = optimizer
    epochss = epochs
    b_size = batch_size
    kfold = StratifiedKFold(n_splits=2, shuffle=True)
    
    classifier = Sequential()
    # Input for wisconsin = 30 nodes, Banknote = 4, Sonar = 60, Abalone = 8, Ionosphere = 33, Pima = 8, Titanic = 26, Heart = 13
    classifier.add(Dense(units=numberOfHiddenLayerNodes,
                        activation=af1, input_dim=dataX.shape[1], use_bias=True))
    # Hidden layers
    for i in range(noOfHiddenLayers):
        classifier.add(Dense(units=numberOfHiddenLayerNodes,
                            activation=af2, use_bias=True))
    # Output layer
    classifier.add(Dense(units=1, activation='sigmoid', use_bias=True))

    # compile the network
    classifier.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m,
                        tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
    for i, (train_index, test_index) in enumerate(kfold.split(dataX, dataY)):
        X_train,X_test = dataX[train_index],dataX[test_index]
        Y_train,Y_test = dataY[train_index],dataY[test_index]
        # Fit data to model
        classifier.fit(X_train, Y_train,
                    batch_size=b_size,
                    epochs=epochss,
                    verbose=0,
                    validation_data=(X_test, Y_test),
                    callbacks=[es, TerminateOnNaN()], use_multiprocessing=True)
        history = classifier.history.history
        last_val = history['val_accuracy'].pop()
        if last_val > 0.1:
            loss, accuracy, f1, precision, recall, mae, rmse = classifier.evaluate(X_test, Y_test, verbose=0)
        else:
            loss, accuracy, f1, precision, recall, mae, rmse = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    if math.isnan(f1):
        f1 = 0

    afs = [af1, af2, af3]
    end = time.time()
    speed = end - start
    out_list = [noOfHiddenLayers, numberOfHiddenLayerNodes, [mae, rmse, accuracy, f1, precision, recall], afs, opti, epochss, b_size, float(speed), last_val]
    que.put(out_list)
    return out_list

def createNN(data, que):
    import psutil
    import os
    global n_folds
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from Metrics import  precision_m, recall_m, f1_m
    # Starting the network
    tf.keras.backend.clear_session()
    
    #X_train, Y_train, X_test, Y_test, X_val, Y_val = data
    dataX, dataY = data
    #dataY = pd.read_csv('./Datasets/WisconsinCancer/y_data.csv', header=None)
    dataX = dataX.to_numpy()
    dataY = dataY.to_numpy()
    noOfHiddenLayers = int(3 * random.random()) + 1
    # The number of hidden layer nodes = 2/3 of the size of the input layer + the size of the output layer (1)
    numberOfHiddenLayerNodes = int(40 * random.random()) + 1
    af1 = activation_functions[random.randint(0, len(activation_functions)-1)]
    af2 = activation_functions[random.randint(0, len(activation_functions)-1)]
    af3 = activation_functions[random.randint(0, len(activation_functions)-1)]
    # Custom
    opti = optimizers[random.randint(0, len(optimizers)-1)]
    #epochss = 3
    epochss = int(50 * random.random()) + 1
    #b_size = 8
    b_size = random.randint(1, 64)
    # Starting the network
    start = time.time()
    ###
    kfold = StratifiedKFold(n_splits=2, shuffle=True)
    
    classifier = Sequential()
    # Input for wisconsin = 30 nodes, Banknote = 4, Sonar = 60, Abalone = 8, Ionosphere = 33, Pima = 8, Titanic = 26, Heart = 13
    classifier.add(Dense(units=numberOfHiddenLayerNodes,
                        activation=af1, input_dim=dataX.shape[1], use_bias=True))
    # Hidden layers
    for _ in range(noOfHiddenLayers):
        classifier.add(Dense(units=numberOfHiddenLayerNodes,
                            activation=af2, use_bias=True))
    # Output layer
    classifier.add(Dense(units=1, activation='sigmoid', use_bias=True))
    classifier.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m,
                        tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
    for i, (train_index, test_index) in enumerate(kfold.split(dataX, dataY)):
        X_train,X_test = dataX[train_index],dataX[test_index]
        Y_train,Y_test = dataY[train_index],dataY[test_index]
        classifier.fit(X_train, Y_train,
                    batch_size=b_size,
                    epochs=epochss,
                    verbose=0,
                    validation_data=(X_test, Y_test),
                    callbacks=[es, TerminateOnNaN()], use_multiprocessing=True)
        history = classifier.history.history
        last_val = history['val_accuracy'].pop()
        epochss = len(history['loss'])
        if last_val > 0.1:
            loss, accuracy, f1, precision, recall, mae, rmse = classifier.evaluate(X_test, Y_test, verbose=0)
        else:
            loss, accuracy, f1, precision, recall, mae, rmse = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    #set f1 only to nan as that is the fitness metric
    if math.isnan(f1):
        f1 = 0

    end = time.time()
    afs = [af1, af2, af3]
    speed = end - start
    out_list = [noOfHiddenLayers, numberOfHiddenLayerNodes, [mae, rmse, accuracy, f1, precision, recall], afs, opti, epochss, b_size, float(speed), last_val]
    que.put(out_list)
    return out_list

def returnStructure(individual, pop):
    return [pop[individual][0], pop[individual][1]]

def initialise_pop(data, pop_size=4):
    
    start = time.time()
    pop = []
    j = 0
    left = pop_size
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Manager() as manager:
        L = manager.list()
        processes = []
        while j < pop_size:
            if left < int(cpu_count):
                cpu_count = left
            for i in range(cpu_count):
                queue1 = multiprocessing.Queue()
                p = multiprocessing.Process(target=createNN, args=(data, queue1))
                p.start()
                pop.append(queue1.get())
                processes.append(p)
                j += 1
                left -= 1
            for p in processes:
                p.join()
            #pop = list(L)
    #print (pop)
    # for i in range(pop_size):
    #     print ('Individual #' + str(i) + ' created.')
    #     hl, nl, Y_pred, afs, opti, speed, val_acc = createNN(data)
    #     pop.append([hl, nl, Y_pred, afs, opti, speed, val_acc])
    #pop.clear()
    end = time.time()
    speed = end - start
    print ('Speed for parallel of '+str(pop_size)+' individuals: ', speed)
    return pop

def retrain_pop(data, pop):
    start = time.time()
    #pop = []
    pop_size = len(pop)
    j = 0
    new_pop = []
    left = pop_size
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Manager() as manager:
        L = manager.list()
        processes = []
        while j < pop_size:            
            if left < int(cpu_count):
                cpu_count = left
            for i in range(cpu_count):
                queue1 = multiprocessing.Queue()
                noOfHiddenLayers, numberOfHiddenLayerNodes, af, optimizer, epochs, batch_size = pop[j]
                p = multiprocessing.Process(target=createAndReturnFitness, args=(noOfHiddenLayers, numberOfHiddenLayerNodes, af, optimizer, epochs, batch_size, data, queue1))
                p.start()
                new_pop.append(queue1.get())
                processes.append(p)
                j += 1
                left -= 1
            for p in processes:
                p.join()

            #pop = list(L)
    # for i in range(pop_size):
    #     print ('Individual #' + str(i) + ' created.')
    #     hl, nl, Y_pred, afs, opti, speed, val_acc = createNN(data)
    #     pop.append([hl, nl, Y_pred, afs, opti, speed, val_acc])
    #pop.clear()
    end = time.time()
    speed = end - start
    #print ('Speed for parallel of '+str(pop_size)+' individuals: ', speed)
    return new_pop    

def printResults(x, y, z):
    if (x == 1):
        x = "Malignant"
    else:
        x = "Benign"
    if (y == 1):
        y = "Malignant"
    else:
        y = "Benign"
    print(z, "actual->", x, ", prediction ->", y)


def test(Y_test, pop):
    # Cycle through the predictions
    for i in range(len(pop)):
        overallFitness = pop[i][2][3]
        pop[i].append(overallFitness)
    return pop[i][8]

# Roulette wheel selection


def roulette_selection(pop, data, cloning_rate):
    import operator
    total_from_set = 0
    sorted_pop = sorted(pop, key=operator.itemgetter(8))
    n = math.ceil(len(pop)*cloning_rate)
    cloned_pop = sorted_pop[:n]
    sorted_pop = sorted_pop[n:]
    max_values = 0
    
    dic = {}
    for i in range(len(sorted_pop)):
        dic[i] = sorted_pop[i]
        max_values += sorted_pop[i][8]
    
    choices = [0.0]
    for i in range(len(dic)):
        try:
            chance = (dic[i][8] / max_values) * 100
        except (ZeroDivisionError):
            chance = 100 / len(pop)
        choices.append(choices[i] + chance)

    #print (choices)
    temp_pop = []
    
    while len(temp_pop) < len(sorted_pop):
        first_spinOfWheel = random.uniform(0.0, 100.0)
        second_spinOfWheel = random.uniform(0.0, 100.0)
        first_chosen_index = 0
        second_chosen_index = 0

        while first_chosen_index == 0:
            for i in range(len(choices)):
                if first_spinOfWheel <= choices[i]:
                    first_chosen_index = i
                    break      
        same = True
        while second_chosen_index == 0 and same == True:
            for i in range(len(choices)):
                if second_spinOfWheel <= choices[i]:
                    second_chosen_index = i
                    if second_chosen_index == first_chosen_index:
                        second_spinOfWheel = random.uniform(0.0, 100.0)
                        second_chosen_index = 1                        
                    else:
                        same = False
                        break            
        child1, child2 = two_child_crossover(dic[first_chosen_index-1], dic[second_chosen_index-1])
        temp_pop.append(child1)
        temp_pop.append(child2)

    new_pop = []
    temp_pop = retrain_pop(data, temp_pop)

    new_pop.extend(cloned_pop)
    new_pop.extend(temp_pop)
    return new_pop

def tournament_selection(Y_test, pop):
    individual1 = random.randint(0, len(pop) - 1)
    individual2 = random.randint(0, len(pop) - 1)
    while individual1 == individual2:
        individual2 = random.randint(0, len(pop) - 1)

    fitness_of_1 = checkFitnessOfIndividual(individual1, Y_test, pop)
    fitness_of_2 = checkFitnessOfIndividual(individual2, Y_test, pop)
    if fitness_of_1 >= fitness_of_2:
        selected = individual1
    else:
        selected = individual2
    return selected


def mutate(data, pop, mut_rate):
    #print ("popIndex from mutate = ", popIndex)
    # grammar is a tuple consisting of:
    # 1. Number of Hidden layers,
    # 2. Number of Nodes per Hidden Layer
    # 3. Af1
    # 4. Af2
    # 5. Af3
    # 6. Optimizer
    # 7. Epochs
    # 8. Batch size
    #[noOfHiddenLayers, numberOfHiddenLayerNodes, [mae, rmse, accuracy, f1, precision, recall], afs, opti, epochss, b_size, float(speed), last_val]
    
    import copy
    temp_pop = copy.deepcopy(pop)
    #unchanged = copy.deepcopy(temp_pop)
    for sublist in temp_pop:
        del sublist[2]
        del sublist[6]
        del sublist[6]
    unchanged = copy.deepcopy(temp_pop)
    for i in range(len(temp_pop)):   
        whichMutation = random.randint(0, len(pop[0]))
        chance = random.randint(0, 100)
        
        if chance <= mut_rate:
            if whichMutation == 0:
                temp_pop[i][whichMutation] = int(3 * random.random()) + 1
            elif whichMutation == 1:
                temp_pop[i][whichMutation] = int(40 * random.random()) + 1
            elif whichMutation == 2 :
                temp_pop[i][2][0] = activation_functions[random.randint(0, len(activation_functions)-1)]
            elif whichMutation == 3:
                temp_pop[i][2][1] = activation_functions[random.randint(0, len(activation_functions)-1)]
            elif whichMutation == 4:
                temp_pop[i][2][2] = activation_functions[random.randint(0, len(activation_functions)-1)]
            elif whichMutation == 5:
                temp_pop[i][whichMutation-2] = optimizers[random.randint(0, len(optimizers)-1)]
            elif whichMutation == 6:
                temp_pop[i][whichMutation-2] = int(50 * random.random()) + 1
            elif whichMutation == 7:
                temp_pop[i][whichMutation-2] = int(64 * random.random()) + 1
        
    retrainable_pop = []
    unchanged_pop = []
    for i in range(len(temp_pop)):
        #has there been a mutation, if so, add it to a list to be retrained
        if temp_pop[i] != unchanged[i]:
            retrainable_pop.append(temp_pop[i])
        else:
            unchanged_pop.append(pop[i])
    

    retrained_pop = retrain_pop(data, retrainable_pop)
    unchanged_pop.extend(retrained_pop)
    return unchanged_pop



def greatest(int1, int2):
    if int1 >= int2:
        return int1
    else:
        return int2


def smallest(int1, int2):
    if int1 <= int2:
        return int1
    else:
        return int2


def two_child_crossover(c1, c2):
    import copy
    # There are 9 possible candidates, this can be treated as the phenotype [0, 1, 2, 3, 4, 5]
    # 0 = Hidden layers
    # 1 = Number of nodes per hidden layers
    # 3 = activation function 1
    # 4 = activation function 2
    # 5 = activation function 3
    # 6 = optimizer
    # 7 = epochs
    # 8 = batch size

    child1 = copy.deepcopy(c1)
    if len(child1) > 6:
        del child1[2]
        del child1[6]
        del child1[6]

    child2 = copy.deepcopy(c2)
    if len(child2) > 6:
        del child2[2]
        del child2[6]
        del child2[6]
    crossover_point = int(len(child1) * random.random())
    for i in range(crossover_point):
        # Dont swap the fitness metrics
        #if i != 2 and i != 7 and i != 8:
        temp_i = child1[i]
        child1[i] = child2[i]
        child2[i] = temp_i
    
    return child1, child2

def pop_crossover(data, pop):
    # There are 9 possible candidates, this can be treated as the phenotype [0, 1, 2, 3, 4, 5]
    # 0 = Hidden layers
    # 1 = Number of nodes per hidden layers
    # 3 = activation function 1
    # 4 = activation function 2
    # 5 = activation function 3
    # 6 = optimizer
    # 7 = epochs
    # 8 = batch size
    crossover_point = int(len(pop[popIndex1]) * random.random())
    for i in range(crossover_point):
        # Dont swap the fitness or weights
        if i != 2 and i != len(pop[popIndex1]):
            temp_i = pop[popIndex1][i]
            pop[popIndex1][i] = pop[popIndex2][i]
            pop[popIndex2][i] = temp_i
    
    pop = retrain_pop(data, pop)
    return pop

def checkFitnessOfIndividual(index, Y_test, pop):
    overallFitness = pop[index][8]
    return overallFitness

def checkAllFitnesses(Y_test):
    fitnessOfAll = test(Y_test)
    for i in range(len(fitnessOfAll)):
        print("Fitness of NN ", i, " = ", fitnessOfAll[i])

def get_guesses_and_fitness(pop, Y_test):
    fitnesses = []
    temp_pop = []
    for i in range(len(pop)):
        overallFitness = pop[i][8]
        fitnesses.append(overallFitness)
        temp_pop.append(pop[i])

    index = fitnesses.index(max(fitnesses))
    fittest_fitnesses = fitnesses[index]
    fittest_guesses = temp_pop[index][2]
    return fittest_guesses, fittest_fitnesses

def whichFittest(pop):
    global highest
    global best_in_pop
    average = 0.0
    for i in range(len(pop)):
        overallFitness = pop[i][8]
        average += overallFitness
        if overallFitness > highest:
            highest = overallFitness
            best_in_pop = pop[i]
    average = average / len(pop)
    return highest, average

def run(data, dataset_name, evo_params, j=0):
    from time import sleep
    from rich.console import Console
    console = Console()
    pop_size = evo_params['population_size']
    mutation_rate = evo_params['mutation_rate']
    cloning_rate = evo_params['cloning_rate']
    max_generations = evo_params['max_generations']
    #mutate_value, crossover_value, generation_count, save_url
    print ('Initialising the population, please stand by...')
    pop = initialise_pop(data, pop_size)
    plot_generation = []
    plot_best_fitness = []
    plot_avg_fitness = []
    output_file = dataset_name+'_GA_p_'+str(pop_size)+'_mr_'+str(mutation_rate)+'_cr_'+str(cloning_rate)+'_mg'
    with console.status("[bold green]Running through generations...") as status:
        for i in range(max_generations+1):
            mutate_rate = mutation_rate
            pop = mutate(data, pop, mutate_rate)
            # # roulette selection takes in the population as an input
            pop = roulette_selection(pop, data, cloning_rate)
            best_fitness, avg_fitness = whichFittest(pop)
            #Every 50th generation, save the fittest network in a file.
            if i % 50 == 0 or best_fitness >= 1.0:
                num = str(i)
                string0 = "Validation accuracy = " + str(best_in_pop[8])
                string00 = "Speed = " + str(best_in_pop[5])
                string1 = "MAE = " + str(best_in_pop[2][0])
                string2 = "Test acc = " + str(best_in_pop[2][2])
                string3 = "RMSE = " + str(best_in_pop[2][1])
                string4 = "Precision = " + str(best_in_pop[2][4])
                try:
                    string5 = "Recall = " + str(best_in_pop[2][5])
                except ValueError:
                    string5 = "ROC AUC Error"
                string6 = "F Measure score = " + str(best_in_pop[2][3])
                string7 = "Individual = " + str(best_in_pop)
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
                if best_fitness >= 1.0:
                    break
            plot_generation.append(i)
            plot_best_fitness.append(best_in_pop[8])  
            plot_avg_fitness.append(avg_fitness)  
            console.log(f"Generation {i} complete...")
        plt.plot(plot_generation, plot_best_fitness, label='Best fitness')
        plt.plot(plot_generation, plot_avg_fitness, label='Average fitness')
        plt.legend(loc='best')
        plt.xlabel('Number of Generations')
        plt.ylabel('Fitness measured in Validation Accuracy', size=12)
        plt.savefig('./Results/Reformatted/'+output_file+'.png')
        plt.close()
        pop.clear()
        best_in_pop.clear()
        global highest
        highest = 0
    completion_message = 'Evolution complete'
    print (completion_message)
    gc.collect()
    return None

def open_data_for_kfold(data_loc):
    dataX = pd.read_csv(data_loc+'/x_data.csv', header=None)
    dataY = pd.read_csv(data_loc+'/y_data.csv', header=None)
    return dataX, dataY

def wrapper(evo_params, data_loc):
    data = open_data_for_kfold(data_loc)
    dataset_name = data_loc.rsplit('/', 1)[-1]
    run(data, dataset_name, evo_params)
    return None

def test_only(weights_file):
    print ('Sorry this function has not been implemented yet')
    return None

if __name__ == '__main__':

    #for j in range(0, 10):
    data = open_dataset('./Datasets/WisconsinCancer')
    j = 1
    run(j, data, 'WisconsinCancer')
    
    # for j in range(0, 10):
    #     data = open_dataset('./Datasets/Titanic')
    #     run(j, data, 'Titanic')
    
    # for j in range(0, 10):
    #     data = open_dataset('./Datasets/Pima')
    #     run(j, data, 'Pima')
    
    # for j in range(0, 10):
    #     data = open_dataset('./Datasets/Sonar')
    #     run(j, data, 'Sonar')
    
    # for j in range(0, 10):
    #     data = open_dataset('./Datasets/Heart')
    #     run(j, data, 'Heart')
    