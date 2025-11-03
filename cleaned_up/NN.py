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
parser.add_argument("--noearlystopping", default=False, action="store_true")
parser.add_argument("--uniform", default=False, action="store_true")
parser.add_argument("--half_data", default=False, action="store_true")

args = parser.parse_args()

if args.half_data:
	N_data = 25000
else:
	N_data = 50000

def to_categorical(Y, N_classes=2):
	Y=np.array(Y,dtype=int)
	return np.eye(N_classes)[Y]

def make_model(activation="relu",hidden=3,inputs=4,lr=1e-3,dropout=0, l1=0, l2 =0, momentum = 0.9, label_smoothing=0):
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

def oned_sample(N, rv):
    x = rv.rvs(N).reshape((N,1))
    y = rv.rvs(N).reshape((N,1))
    return np.concatenate((x,y), axis=1)

if args.uniform: 
    rv = stats.uniform(loc=-2, scale=4)
else:
	rv = multivariate_normal([0,0], [[1,0],[0,1]])
if not os.path.exists(args.directory):
	os.makedirs(args.directory)

BT_test_preds = np.zeros((args.runs, N_data//2))
data_test_preds = np.zeros((args.runs, N_data//2))
BT_train_preds = np.zeros((args.runs, N_data//2))
data_train_preds = np.zeros((args.runs, N_data//2))
if not args.noearlystopping:
	BT_valtrainsplitlabels = np.zeros((args.runs, N_data//2))
	data_valtrainsplitlabels = np.zeros((args.runs, N_data//2))
split_labels = np.concatenate((np.zeros(N_data//2), np.ones(N_data//2)))

for i in range(args.start_runs, args.start_runs+args.runs):
	if args.uniform:
		data = oned_sample(int(2*N_data), rv)
	else:
		data = rv.rvs(int(2*N_data))
	X_train, X_test =np.array_split(data,2)

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
		callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)]
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


	data_test, BT_test = np.array_split(X_test, 2)
	BT_test_preds[i-args.start_runs] = model.predict(BT_test)[:,1]
	data_test_preds[i-args.start_runs] = model.predict(data_test)[:,1]
	BT_train_preds[i-args.start_runs] = model.predict(X_train[Y_train[:,1]==0])[:,1]
	data_train_preds[i-args.start_runs] = model.predict(X_train[Y_train[:,1]==1])[:,1]
	if not args.noearlystopping:
		BT_valtrainsplitlabels[i-args.start_runs] = split_labels[Y_train[:,1]==0]
		data_valtrainsplitlabels[i-args.start_runs] = split_labels[Y_train[:,1]==1]

np.save(args.directory+"runs/run"+str(args.start_runs)+"_test_data_preds.npy", data_test_preds)
np.save(args.directory+"runs/run"+str(args.start_runs)+"_test_BT_preds.npy", BT_test_preds)
np.save(args.directory+"runs/run"+str(args.start_runs)+"_train_data_preds.npy", data_train_preds)
np.save(args.directory+"runs/run"+str(args.start_runs)+"_train_BT_preds.npy", BT_train_preds)
if not args.noearlystopping:	
	np.save(args.directory+"runs/run"+str(args.start_runs)+"_splitlabels_BT.npy", BT_valtrainsplitlabels)
	np.save(args.directory+"runs/run"+str(args.start_runs)+"_splitlabels_data.npy", data_valtrainsplitlabels)