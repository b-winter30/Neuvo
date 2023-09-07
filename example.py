from neuvo import NeuvoBuilder
import tensorflow as tf
import numpy as np
import pandas as pd

def load_cnn_data(dir_path):
    if 'cifar' in dir_path:
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        df = pd.DataFrame(list(zip(x_train, y_train)), columns =['Image', 'label']) 
        val = df.sample(frac=0.02)
        X_train = np.array([ i for i in list(val['Image'])])
        Y_train = np.array([ [i[0]] for i in list(val['label'])])
        test = pd.DataFrame(list(zip(x_test, y_test)), columns =['Image', 'label']) 
        val_test = test.sample(frac=0.02)
        X_test = np.array([ i for i in list(val_test['Image'])])
        Y_test = np.array([ [i[0]] for i in list(val_test['label'])])
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
    dataX = X
    dataY = Y
    return [dataX, dataY]

def load_1d_data(dir_path):
    dataX = pd.read_csv('../neuro_obj/Datasets/'+dir_path+'/x_data.csv', header=None)
    dataY = pd.read_csv('../neuro_obj/Datasets/'+dir_path+'/y_data.csv', header=None)
    
    return [dataX.to_numpy(), dataY.to_numpy()]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", default="WisconsinCancer", help="Dataset location")
    parser.add_argument("-t", "--type", default='ga',  help="Type of evolutionary algorithm ga/ge")
    parser.add_argument("-e", "--eco", action='store_true', help="Ecologoical mode (True/False)")
    parser.add_argument("-ne", "--no-eco", dest='eco', action='store_false')
    parser.set_defaults(eco=False)
    args = vars(parser.parse_args())
    
    data = load_1d_data(args["dataset"])
    Neuvo = NeuvoBuilder(type=args["type"], eco=args["eco"])

    #Neuvo.grammar_file='basic_grammar.txt'
    Neuvo.selection='Tournament'
    Neuvo.crossover_method='two_point'
    Neuvo.population_size=3
    Neuvo.mutation_rate=1.0
    Neuvo.cloning_rate=0.33
    Neuvo.max_generations=3
    Neuvo.verbose=0

    Neuvo.dataset_name = args["dataset"]
    Neuvo.load_data(data)
    Neuvo.set_fitness_function('val_acc_x_f1')
    Neuvo.initialise_pop(elite_mode=True)
    Neuvo.run(plot=True, verbose=True)
