import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
import time
from Metrics import f1_m, precision_m, recall_m
import pandas as pd
import tensorflow as tf
import math

def build_basic_ann():
    dataX = pd.read_csv('./Datasets/Pima/x_data.csv', header=None)
    dataY = pd.read_csv('./Datasets/Pima/y_data.csv', header=None)
    dataX = dataX.to_numpy()
    dataY = dataY.to_numpy()
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(units=8, activation='relu', input_dim=8, use_bias=True))
    model.add(Dense(units=8, activation='relu', use_bias=True))
    model.add(Dense(units=8, activation='relu', use_bias=True))
    model.add(Dense(units=8, activation='relu', use_bias=True))
    model.add(Dense(units=1, activation='sigmoid', use_bias=True))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m, MeanAbsoluteError(), RootMeanSquaredError()])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
    num_folds = 2
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3)
    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True)
    start = time.time()
    loss, accuracy, f1, precision, recall, mae, rmse = (0.0,)*7
    for i, (train_index, test_index) in enumerate(kfold.split(dataX, dataY)):
        X_train,X_test = dataX[train_index],dataX[test_index]
        Y_train,Y_test = dataY[train_index],dataY[test_index]
        model.fit(X_train, Y_train, batch_size=4, epochs=20,
                        verbose=0, validation_data=(X_test, Y_test), callbacks=[es, TerminateOnNaN()])
        history = model.history.history
        last_val = history['val_accuracy'].pop()
        #if last_val > 0.1:
        los, acc, f, prec, rec, ma, rms = model.evaluate(X_test, Y_test, verbose=0)
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

def plot():

    res = [0.4281, 0.3778, 0.3644, 0.3931, 0.3917, 0.3942, 0.3793, 0.3813, 0.3756, 0.3609]
    speed = [14.480, 7.9025, 3.4425, 3.0104, 4.1389, 4.7615, 2.1565, 2.1400, 1.9854, 1.9395]

    new_x, new_y = zip(*sorted(zip(res, speed)))

    plt.plot(new_x, new_y, marker='o', markerfacecolor='red', color='black', linestyle='')
    plt.xlabel('Validation accuracy x F1 Score')
    plt.ylabel('Training and testing speed (s)')
    plt.show()

if __name__ == '__main__':
    plot()
