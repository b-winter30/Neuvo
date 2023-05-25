import neuro
import tensorflow as tf
from Metrics import f1_m, precision_m, recall_m
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import time

def run():
	dataX = pd.read_csv('./Datasets/WisconsinCancer/x_data.csv', header=None)
	dataY = pd.read_csv('./Datasets/WisconsinCancer/y_data.csv', header=None)
	dataX = dataX.to_numpy()
	dataY = dataY.to_numpy()
	model = Sequential()
	model.add(Dense(units=47, activation='tanh', input_dim=dataX.shape[1], use_bias=True))
	model.add(Dense(units=47, activation='softplus', use_bias=True))
	model.add(Dense(units=1, activation='tanh', use_bias=True))
	model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m,
					MeanAbsoluteError(), RootMeanSquaredError()])

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
		model.fit(X_train, Y_train, batch_size=30, epochs=5,
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
	print (metrics)

if __name__ == '__main__':
	for i in range(10):
		run()