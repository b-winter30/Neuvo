from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
from math import sqrt
from tensorflow.keras import backend as K
import numpy as np

def get_mae(list_of_guesses, actual_list):
    return mean_absolute_error(actual_list, list_of_guesses)

def get_mse(list_of_guesses, actual_list):
    return mean_squared_error(actual_list, list_of_guesses)

def get_rmse(list_of_guesses, actual_list):
    return sqrt(mean_squared_error(actual_list, list_of_guesses))

def get_precision(list_of_guesses, actual_list):
    return precision_score(actual_list, list_of_guesses, average='binary')

def get_recall(list_of_guesses, actual_list):
    return recall_score(actual_list, list_of_guesses, average='binary')

def get_standard_deviation(a_list):
    return np.std(a_list)

def get_standard_deviation_of_two_lists(list_of_guesses, actual_list):
    A = get_standard_deviation(list_of_guesses)
    B = get_standard_deviation(actual_list)
    return np.sum(np.abs(B - A))

def get_rocauc(list_of_guesses, actual_list):
    return roc_auc_score(actual_list, list_of_guesses)

def get_f_measure(list_of_guesses, actual_list):
    return f1_score(actual_list, list_of_guesses)

def get_conf_matrix(list_of_guesses, actual_list):
    return confusion_matrix(actual_list, list_of_guesses).ravel()

def get_f_measure_parallel(list_of_guesses, actual_list, shared_variable, individual):
    shared_variable[individual] = f1_score(actual_list, list_of_guesses)
    #shared_variable.append(f1_score(actual_list, list_of_guesses))
    
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
