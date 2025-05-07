import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 
import scipy.stats as stats

import tensorflow as tf
from tensorflow import keras
models = keras.models
layers = keras.layers
regularizers = keras.regularizers
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--start_runs", default=0, type=int)
parser.add_argument("--runs", default=100, type=int)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--trainistest", default=False, action="store_true")
args = parser.parse_args()

def to_categorical(Y, N_classes=2):
	Y=np.array(Y,dtype=int)
	return np.eye(N_classes)[Y]

def make_model(activation="relu",hidden=3,inputs=4,lr=1e-3,dropout=0.1, l1=0, l2 =0, momentum = 0.9, label_smoothing=0):
	model = models.Sequential()
	model.add(layers.Dense(64,input_shape=(inputs,)))
	for i in range(hidden-1):
		if activation =="relu":
			model.add(layers.ReLU())
		elif activation == "leaky":
			model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Dropout(dropout))
		model.add(layers.Dense(64,kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
		model.add(layers.ReLU())
	model.add(layers.Dense(2, activation="softmax"))

	loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

	model.compile(
		loss=loss,
		optimizer=keras.optimizers.Adam(lr, beta_1=momentum),
		metrics=["accuracy"],
	)

	return model

rv = multivariate_normal([0,0], [[1,0],[0,1]])
if not os.path.exists(args.directory):
	os.makedirs(args.directory)

for i in range(args.start_runs, args.start_runs+args.runs):
	
	direc_run=args.directory+"run"+str(i)+"/"
	if not os.path.exists(direc_run):
		os.makedirs(direc_run)
		
	if args.trainistest:
		data = rv.rvs(200000)
		data_train, BT_train = np.array_split(data, 2)
		data_test = data_train
		BT_test = BT_train
	else: 
		data = rv.rvs(400000)
		data_train, data_test, BT_train, BT_test = np.array_split(data, 4)

	X_train = np.concatenate((data_train, BT_train), axis=0)
	Y_train = to_categorical(np.append(np.ones(len(data_train)), np.zeros(len(BT_train))))
	inds = np.arange(len(X_train))
	np.random.shuffle(inds)
	X_train = X_train[inds]
	Y_train = Y_train[inds]

	model = make_model(inputs=2)

	earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

	results = model.fit(
		X_train,
		Y_train,
		batch_size=1024,
		epochs=100,
		shuffle=True,
		verbose=2,
		validation_split=0.5,
		callbacks=[earlyStopping],
	)

	np.save(direc_run+'classifier_history.npy', results.history)
	np.save(direc_run+"data_preds.npy", model.predict(data_test))
	np.save(direc_run+"BT_preds.npy", model.predict(BT_test))
	model.save(direc_run+"model.keras")