import collections
import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
from sklearn.model_selection import StratifiedKFold
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
import Grammatical_Neuro
import re
populationCount = 2
highest = 0
best_in_pop = []
type = ''
bitstring = []
final_grammar_list = ""
basic_ops = {
    '+' : tf.add,
    '-' : tf.subtract,
    '*' : tf.multiply,
    '/' : tf.divide
}
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


def open_dataset(directory):
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    dataX = pd.read_csv(directory+'/x_data.csv', header=None)
    dataY = pd.read_csv(directory+'/y_data.csv', header=None)
    # train is now 75% of the entire data set
    X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, shuffle=True)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=True) 
    return (X_train, Y_train, X_test, Y_test, X_val, Y_val)

def recursion(grammar, genome, step, start_bool):
    global phenotype
    returnable_pheno = ""
    if start_bool:
        try:
            phenotype == ""
        except (NameError):
            phenotype = ""
        step == 0
        codon = grammar['start'][bitstring[step] % len(grammar['start'])][0]
        genome.append(codon)
        step += 1
        recursion(grammar, genome, step, False)
        ###
        #Reason for the return statements: "But returning a result from a function always returns
        #it to the direct caller of this function. It doesn't jump immediately out through several calls;"
    else:
        #Sanity check for wrapping
        if step == len(bitstring):
            step = 0
        if len(genome) > 0:
            if genome[0] == '(' or genome[0] == ')' or genome[0] == ',' or genome[0] in basic_ops:
                phenotype = phenotype + str(genome[0])
                genome.pop(0)
                recursion(grammar, genome, step, False)
                return None
            codon = grammar[genome[0]][bitstring[step] % len(grammar[genome[0]])]
            #Non-Terminal
            if tf.is_tensor(codon[0]):
                if codon[0].ref() in grammar: 
                    genome = codon + genome  
                    genome.pop(0)
                    step += 1
                    recursion(grammar, genome, step, False)
                    return None
                #Terminal check
                else:
                    phenotype = phenotype + str(codon[0])
                    genome.pop(0)
                    step += 1
                    recursion(grammar, genome, step, False)
                    return None
            else: #
                if codon[0] in grammar:  
                    genome.pop(0)
                    genome = codon + genome
                    step += 1
                    recursion(grammar, genome, step, False)
                    return None
                #Terminal check
                else:
                    phenotype = phenotype + str(codon[0])
                    genome.pop(0)
                    step += 1
                    
                    if len(codon) != 1:
                        t = codon[1:]
                        genome = t + genome
                    else:
                        codon = []
                    recursion(grammar, genome, step, False)
                    return None
    returnable_pheno = phenotype 
    codon = []
    return returnable_pheno

acti_calls = 0

def work_out(text, tensor):
    global phenotype
    text = text[:500]
    global final_grammar_list
    #final_grammar_list = text
    new_text = text
    count = 1
    for m in re.finditer('\)', text[:-1]):
        i = m.start()
        if (text[i+1] not in basic_ops)and text[i+1] != ')' and text[i+1] != ',':
            #new_text = new_text[:i+count] + "+" + new_text[i+count:]
            new_text = new_text[:i+count]
            count += 1
    final_grammar_list = new_text
    try:
        eval_num = eval(new_text)
    except (ZeroDivisionError):
        eval_num = tf.math.multiply(0.0, tensor)
        final_grammar_list = '0.0*tensor'
    except (SyntaxError, NameError):
        print ('p')
        print (new_text)
        print ('phenotype = ', phenotype)
        print ('final_grammar_list = ', final_grammar_list)
    except (MemoryError):
        print ('Genotype does not compile.')
        eval_num = tensor*0.0
        final_grammar_list = 'tensor*0.0'
    #Shape is () - example phenotype for this error = tf.math.reduce_sum(tf.math.abs(tensor))
    if type(eval_num) == tuple:
        print ("eval_num is a tuple = ", eval_num)
        eval_num = tf.math.multiply(tensor,eval_num)
    #Initially this code was that if it didn't find a tensor it would perform: tensor*final_grammar_list. However,
    #This means it could make every result the same in the tensor and still be fittest (in linearly seperable problems). We need to kill the phenotype
    #if tensor is not found. And semi-force a tensor to be plucked from the genotype.
    if type(eval_num) == float or 'tensor' not in new_text:
        eval_num = tf.math.multiply(0.0, tensor)
        final_grammar_list == '0.0*tensor'
    if eval_num.shape == ():
        eval_num = eval_num*tensor

    new_text = ""
    phenotype = ""
    return eval_num
    #return tensor

def input_activation(x):
    #This way, it will only go through the workout function once.
    try:
        
        if final_grammar_list == "":
            new_num = recursion(grammar, [], 0, True)
            new_num_array = work_out(new_num, x)
        else:
            new_num_array = work_out(final_grammar_list, x)
    except (RecursionError):
        #print ('bitstring from error: ', bitstring)
        new_num_array = work_out('0.0*tensor', x)
    #rounded_tens = tf.math.maximum(0.0, new_num_array)
    #rounded_tens = my_tf_round(new_num_array, 8)
    rounded_tens = new_num_array
    return rounded_tens

def my_tf_round(x, decimals = 0):
    if type(x) == list:
        x = x[0]
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    rounded_tens = tf.round(tf.math.divide_no_nan((tf.math.multiply(x, multiplier)), multiplier))
    #Check for infs
    rounded_tens = tf.where(tf.math.is_inf(rounded_tens), tf.zeros_like(rounded_tens), rounded_tens)
    return rounded_tens

def createAndReturnFitness(bitstrings, data, queue):
    global phenotype
    global final_grammar_list
    global bitstring
    #Read data
    dataX, dataY = data
    dataX = dataX.to_numpy()
    dataY = dataY.to_numpy()
    # Starting the network
    tf.keras.backend.clear_session()
    start = time.time()
    
    kfold = StratifiedKFold(n_splits=2, shuffle=True)
    if type.lower() == 'af':
        noOfHiddenLayers = 3
        numberOfHiddenLayerNodes = 8
        afs = []
        opti = 'Adamax'
        epochss = 25
        b_size = 4
        classifier.add(Dense(units=numberOfHiddenLayerNodes,
                    activation=input_activation, input_dim=dataX.shape[1], use_bias=True))
        afs.append(final_grammar_list)
        final_grammar_list = ""
        # Hidden layers
        try:
            for j in range(1, noOfHiddenLayers):
                bitstring = af_genomes[j]
                classifier.add(Dense(units=numberOfHiddenLayerNodes,
                            activation=input_activation, use_bias=True))
                afs.append(final_grammar_list)
                final_grammar_list = ""
        except IndexError:
            pass
    else:
        genome = Grammatical_Neuro.create_genome(bitstrings[0])
        af_genomes = []
        afs = []
        for i in range(0, genome[0]+1):
            temp_af = [int(random.random() * 100) for _ in range(16)]
            af_genomes.append(temp_af)
        bitstring = af_genomes[0]
        noOfHiddenLayers = genome[0]
        numberOfHiddenLayerNodes = genome[1]
        opti = genome[3]
        epochss = genome[4]
        b_size = genome[5]
        classifier = Sequential()
        if type.lower() == 'both':
            # Input for wisconsin = 30 nodes, Banknote = 4, Sonar = 60, Abalone = 8, Ionosphere = 33, Pima = 8, Titanic = 26, Heart = 13
            classifier.add(Dense(units=numberOfHiddenLayerNodes,
                        activation=input_activation, input_dim=dataX.shape[1], use_bias=True))
            afs.append(final_grammar_list)
            final_grammar_list = ""
            # Hidden layers
            try:
                for j in range(1, noOfHiddenLayers):
                    bitstring = af_genomes[j]
                    classifier.add(Dense(units=numberOfHiddenLayerNodes,
                                activation=input_activation, use_bias=True))
                    afs.append(final_grammar_list)
                    final_grammar_list = ""
            except IndexError:
                pass
        else:
            afs = ['relu', 'relu', 'relu']
            classifier.add(Dense(units=numberOfHiddenLayerNodes,
                        activation=afs[0], input_dim=dataX.shape[1], use_bias=True))
            # Hidden layers
            try:
                for j in range(1, noOfHiddenLayers):
                    classifier.add(Dense(units=numberOfHiddenLayerNodes,
                                activation=afs[i], use_bias=True))
            except IndexError:
                pass
    # Output layer
    classifier.add(Dense(units=1, activation='sigmoid', use_bias=True))
    afs.append('sigmoid')
    # compile the network
    classifier.compile(optimizer=opti, loss='binary_crossentropy',
    metrics=['accuracy',f1_m,precision_m, recall_m,
            tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3)
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
        epochss = len(history['loss'])
        # For binary output that is 1 or 0
        # Cancer
        if last_val > 0.1:
            loss, accuracy, f1, precision, recall, mae, rmse = classifier.evaluate(X_test, Y_test, verbose=0)
        else:
            loss, accuracy, f1, precision, recall, mae, rmse = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    #set f1 only to nan as that is the fitness metric
    if math.isnan(f1):
        f1 = 0

    geno = af_genomes
    geno.insert(0, bitstrings[0])

    end = time.time()
    speed = end - start
    individual = [noOfHiddenLayers, numberOfHiddenLayerNodes, [mae, rmse, accuracy, f1, precision, recall], 
                afs, opti, epochss, b_size, float(speed), last_val, geno]
    final_grammar_list = ""
    bitstring == ""
    phenotype = ""
    queue.put(individual)
    return individual

def createNN(data, que):
    global phenotype
    global final_grammar_list
    global bitstring
    dataX, dataY = data
    dataX = dataX.to_numpy()
    dataY = dataY.to_numpy()
    tf.keras.backend.clear_session()
    local_type = type
    # Starting the network
    start = time.time()
    kfold = StratifiedKFold(n_splits=2, shuffle=True)
    
    classifier = Sequential()
    #If it is 'both'\
    if local_type.lower() == 'af':
        noOfHiddenLayers = 3
        numberOfHiddenLayerNodes = 8
        opti = 'Adamax'
        epochss = 25
        afs = []
        b_size = 4
        classifier.add(Dense(units=numberOfHiddenLayerNodes,
                        activation=input_activation, input_dim=dataX.shape[1], use_bias=True))
        afs.append(final_grammar_list)
        final_grammar_list = ""
        # Hidden layers
        try:
            for j in range(1, noOfHiddenLayers):
                bitstring = af_genomes[j]
                classifier.add(Dense(units=numberOfHiddenLayerNodes,
                            activation=input_activation, use_bias=True))
                afs.append(final_grammar_list)
                final_grammar_list = ""
        except IndexError:
            pass
    else:
        passed_geno = [int(random.random() * 100) for _ in range(16)]
        genome = Grammatical_Neuro.create_genome(passed_geno) #This should be called phenotype
        af_genomes = []
        for i in range(0, genome[0]+1):
            temp_af = [int(random.random() * 100) for _ in range(16)]
            af_genomes.append(temp_af)
        bitstring = af_genomes[0]
        noOfHiddenLayers = genome[0]
        numberOfHiddenLayerNodes = genome[1]
        afs = []
        opti = genome[3]
        epochss = genome[4]
        b_size = genome[5]
        if local_type.lower() == 'both':
            # Input for wisconsin = 30 nodes, Banknote = 4, Sonar = 60, Abalone = 8, Ionosphere = 33, Pima = 8, Titanic = 26, Heart = 13
            classifier.add(Dense(units=numberOfHiddenLayerNodes,
                        activation=input_activation, input_dim=dataX.shape[1], use_bias=True))
            afs.append(final_grammar_list)
            final_grammar_list = ""
            # Hidden layers
            try:
                for j in range(1, noOfHiddenLayers):
                    bitstring = af_genomes[j]
                    classifier.add(Dense(units=numberOfHiddenLayerNodes,
                                activation=input_activation, use_bias=True))
                    afs.append(final_grammar_list)
                    final_grammar_list = ""
            except IndexError:
                pass
        else:
            afs = ['relu', 'relu', 'relu']
            classifier.add(Dense(units=numberOfHiddenLayerNodes,
                        activation=afs[0], input_dim=dataX.shape[1], use_bias=True))
            # Hidden layers
            try:
                for j in range(1, noOfHiddenLayers):
                    classifier.add(Dense(units=numberOfHiddenLayerNodes,
                                activation=afs[i], use_bias=True))
            except IndexError:
                pass
    # Output layer
    classifier.add(Dense(units=1, activation='sigmoid', use_bias=True))
    afs.append('sigmoid')
    classifier.compile(optimizer=opti, loss='binary_crossentropy',
    metrics=['accuracy',f1_m,precision_m, recall_m,
            tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3)
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
    
    geno = af_genomes
    geno.insert(0, passed_geno)
    speed = end - start
    out_list = [noOfHiddenLayers, numberOfHiddenLayerNodes, [mae, rmse, accuracy, f1, precision, recall],
                afs, opti, epochss, b_size, float(speed), last_val, geno]
    #population.append(out_list)
    final_grammar_list = ""
    bitstring == ""
    phenotype = ""
    print ('individual created: ', out_list)
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
    if pop_size > 0:
        cpu_count = multiprocessing.cpu_count()
        with multiprocessing.Manager() as manager:
            L = manager.list()
            processes = []
            while j < pop_size:            
                if left < int(cpu_count):
                    cpu_count = left
                for i in range(cpu_count):
                    queue1 = multiprocessing.Queue()
                    genomes = pop[j]
                    p = multiprocessing.Process(target=createAndReturnFitness, args=(genomes, data, queue1))
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


# Print the actual value, prediction, and if they match or not
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
    import math
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
###
# Mutate 1 quarter of the genome
###
def mutate(data, pop, mut_rate):
    import copy
    from Grammatical_Neuro import flatten_list
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
    temp_pop = copy.deepcopy(pop)
    #unchanged = copy.deepcopy(temp_pop)
    for sublist in temp_pop:
        del sublist[0:9]
    unchanged = copy.deepcopy(temp_pop)
    index_of_genomes = 0
    for i in range(len(temp_pop)):   
        
        chance = random.randint(0, 100)
        for j in range(len(temp_pop[i][index_of_genomes])):
            whichMutation = random.randint(0, len(temp_pop[i][index_of_genomes][j])-1)
            if chance <= mut_rate:
                temp_pop[i][index_of_genomes][j][whichMutation] = int(random.random() * 100)#Change this to 2000 to check that it is working
         
    retrainable_pop = []
    unchanged_pop = []
    for i in range(len(temp_pop)):
        #has there been a mutation, if so, add it to a list to be retrained
        if temp_pop[i] != unchanged[i]:
            #print ('needs to be retrained')
            retrainable_pop.append(temp_pop[i])
        else:
            unchanged_pop.append(pop[i])

    retrained_pop = retrain_pop(data, flatten_list(retrainable_pop))
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
    from Grammatical_Neuro import flatten_list
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
    if len(child1) > 1:
        del child1[0:-1]
        child1 = flatten_list(child1)
    child2 = copy.deepcopy(c2)
    if len(child2) > 1:
        del child2[0:-1]
        child2 = flatten_list(child2)
    if len(child1) <= len(child2):
        for i in range(len(child1)):
            crossover_point = random.randint(0, len(child1[i])-1)
            for j in range(len(child1[i])):
                if j <= crossover_point:
                    temp_i = child1[i][j]
                    child1[i][j] = child2[i][j]
                    child2[i][j] = temp_i
    else:
        for i in range(len(child2)):
            crossover_point = random.randint(0, len(child2[i])-1)
            for j in range(len(child2[i])):
                if j <= crossover_point:
                    temp_i = child2[i][j]
                    child2[i][j] = child1[i][j]
                    child1[i][j] = temp_i
    
    return child1, child2

def single_crossover(pop, popIndex1, popIndex2, data):
    for i in range(0, len(pop[popIndex1][8])-1):

        one_crossover_point = round(
            random.randint(0, len(pop[popIndex1][8][i])-1))
        for j in range(0, one_crossover_point):
            temp_j = pop[popIndex1][8][i][j]
            pop[popIndex1][8][i][j] = pop[popIndex2][8][i][j]
            pop[popIndex2][8][i][j] = temp_j
            #j += i
    child1 = createAndReturnFitness(pop[popIndex1][8], data)
    child2 = createAndReturnFitness(pop[popIndex2][8], data)
    pop[popIndex1] = child1
    pop[popIndex2] = child2
    return pop

def double_crossover(pop, popIndex1, popIndex2, data):
    half = round(len(pop[popIndex1][8]) / 2)
    one_crossover_point = random.randint(0, half)
    two_crossover_point = random.randint(
        one_crossover_point, len(pop[popIndex1][8]))
    for i in range(one_crossover_point, two_crossover_point):
        temp_i = pop[popIndex1][8][i]
        pop[popIndex1][8][i] = pop[popIndex2][8][i]
        pop[popIndex2][8][i] = temp_i
        i += i
    child1 = createAndReturnFitness(pop[popIndex1][8], data)
    child2 = createAndReturnFitness(pop[popIndex2][8], data)
    pop[popIndex1] = child1
    pop[popIndex2] = child2
    return pop

def checkFitnessOfIndividual(pop, index):
    overallFitness = pop[index][2]
    return overallFitness

def checkAllFitnesses(Y_test):
    fitnessOfAll = test(Y_test)
    for i in range(len(fitnessOfAll)):
        print("Fitness of NN ", i, " = ", fitnessOfAll[i])

'''
For any given population, return the fittest individuals guesses and fitness
'''
def get_guesses_and_fitness(pop, Y_test):
    fitnesses = []
    temp_pop = []
    for i in range(len(pop)):
        overallFitness = Metrics.get_f_measure(pop[i][2], Y_test)
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
        average = overallFitness
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
    output_file = dataset_name+'_GE_p_'+str(pop_size)+'_mr_'+str(mutation_rate)+'_cr_'+str(cloning_rate)+'_mg_'+str(type)
    with console.status("[bold green]Running through generations...") as status:
        for i in range(max_generations+1):
            mutate_rate = mutation_rate
            pop = mutate(data, pop, mutate_rate)
            # roulette selection takes in the population as an input
            pop = roulette_selection(pop, data, cloning_rate)
            best_fitness, avg_fitness = whichFittest(pop)
            # Every 50th generation, save the fittest network in a file.
            if i % 50 == 0 or best_fitness >= 1.0:
                num = str(i)
                string0 = "Validation accuracy = " + str(best_in_pop[8])
                string00 = "Speed = " + str(best_in_pop[7])
                string1 = "MAE = " + str(best_in_pop[2][0])
                string2 = "Test acc = " + str(best_in_pop[2][2])
                string3 = "RMSE = " + str(best_in_pop[2][1])
                string4 = "Precision = " + str(best_in_pop[2][4])
                stringgeno = "Genotype = " + str(best_in_pop[9])
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
                    fd.write(stringgeno + "\n")
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

def wrapper(evo_params, data_loc, local_type):
    global type
    type = local_type
    data = open_data_for_kfold(data_loc)
    dataset_name = data_loc.rsplit('/', 1)[-1]
    run(data, dataset_name, evo_params)
    return None

def test_only(evo_params, data_loc, weights_file):
    print ('Not yet implemented')
    return None

# Mutate Rate, Crossover Rate, Generation Count
if __name__ == '__main__':

    data = open_dataset('./Datasets/WisconsinCancer')
    
    #for j in range(0, 10):
    run(1, data, 'WisconsinCancer')
    #[36, 62, 92, 15, 12, 26, 25, 31, 50, 2, 30, 97, 5, 22, 97, 15, 72, 81, 3, 49, 66, 83, 11, 55, 51, 82, 75, 36, 44, 49, 46, 97, 61, 11, 89, 53, 73, 65, 15, 63, 99, 97, 53, 14, 33, 70, 35, 91, 46, 80]

    # data = open_dataset('./Datasets/Titanic')
    # for j in range(0, 10):
    #     run(j, data, 'Titanic')
    
    # data = open_dataset('./Datasets/Pima')
    # for j in range(0, 10):
    #     run(j, data, 'Pima')
    
    # data = open_dataset('./Datasets/Sonar')
    # for j in range(0, 10):
    #     run(j, data, 'Sonar')
    
    # data = open_dataset('./Datasets/Heart')
    # for j in range(0, 10):
    #     run(j, data, 'Heart')
