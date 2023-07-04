from neuro import NeuvoBuilder
import tensorflow as tf
import numpy as np
import pandas as pd

def load_cnn_data(dir_path):
    xs = []
    ys = []
    if 'cifar' in dir_path:
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
        X = np.concatenate((x_train, x_test), axis=0)
        Y = np.concatenate((y_train, y_test), axis=0)
    dataX = X
    dataY = Y
    return [dataX, dataY]

def load_1d_data(dir_path):
    dataX = pd.read_csv('../Datasets/'+dir_path+'/x_data.csv', header=None)
    dataY = pd.read_csv('../Datasets/'+dir_path+'/y_data.csv', header=None)
    
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

    Neuvo.selection='Tournament'
    Neuvo.population_size=3
    Neuvo.mutation_rate=0.01
    Neuvo.cloning_rate=0.3
    Neuvo.max_generations=2
    Neuvo.verbose=0

    Neuvo.dataset_name = args["dataset"]
    Neuvo.load_data(data)
    Neuvo.set_fitness_function('val_acc_x_f1')
    Neuvo.initialise_pop(insertions=insertions)
    Neuvo.run(plot=True)
