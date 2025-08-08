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
parser.add_argument("--folds", default=5, type=int)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--noearlystopping", default=False, action="store_true")


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

	X = rv.rvs(50000)
	data, BT = np.array_split(X,2)
	data = np.array_split(data, args.folds)
	BT = np.array_split(BT, args.folds)

	data_preds = np.zeros((5,5000))
	samples_preds = np.zeros((5,5000))

	for k in range(args.folds):
		inds = np.roll(np.array(range(5)), k)
		X_train = np.concatenate((data[inds[0]], BT[inds[0]]))
		print(X_train.shape)
		Y_train = np.concatenate((np.ones(len(data[inds[0]])), np.zeros(len(BT[inds[0]]))))
		for j in range(1, args.folds-1):
			X_train = np.concatenate((X_train, data[inds[j]], BT[inds[j]]))
			Y_train = np.concatenate((Y_train, np.ones(len(data[inds[j]])), np.zeros(len(BT[inds[j]]))))
		print(X_train.shape)
		data_test = data[inds[-1]]
		BT_test = BT[inds[-1]]

		inds = np.arange(len(X_train))
		np.random.shuffle(inds)
		X_train = X_train[inds]
		Y_train = to_categorical(Y_train[inds])

		Y_train = np.append(np.ones(len(X_train)//2), np.zeros(len(X_train)//2))
		np.random.shuffle(Y_train)
		Y_train = to_categorical(Y_train)

		model = make_model(inputs=2)
		
		if args.noearlystopping:
			results = model.fit(
					X_train,
					Y_train,
					batch_size=1024,
					epochs=100,
					shuffle=True,
					verbose=2,
				)
		else: 
			callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')]
			val_frac = 0.5
			results = model.fit(
					X_train,
					Y_train,
					batch_size=1024,
					epochs=100,
					shuffle=True,
					verbose=2,
					validation_split=val_frac,
					callbacks=callbacks,
				)
			
		data_preds[k] = model.predict(data_test)[:,1]
		samples_preds[k] = model.predict(BT_test)[:,1]

	np.save(direc_run+"test_data_preds.npy", data_preds)
	np.save(direc_run+"test_BT_preds.npy", samples_preds)