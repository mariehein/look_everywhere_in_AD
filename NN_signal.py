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
parser.add_argument("--runs", default=10, type=int)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--N_sig", default=100, type=int)
parser.add_argument("--N_bkg", default=100000, type=int)
parser.add_argument("--dimensions", default=2, type=int)
parser.add_argument("--signal_position", required=True, type=float)
parser.add_argument("--signal_width", required=True, type=float)
args = parser.parse_args()
print(args)
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

rv_bkg = multivariate_normal(np.zeros(args.dimensions), np.diag(np.ones(args.dimensions)))
rv_sig = multivariate_normal(np.ones(args.dimensions)*args.signal_position, np.diag(np.ones(args.dimensions))*args.signal_width)

if not os.path.exists(args.directory):
	os.makedirs(args.directory)

np.save(args.directory+"signal.npy", rv_sig.rvs(10000))
np.save(args.directory+"background.npy", rv_bkg.rvs(10000))

for i in range(args.start_runs, args.start_runs+args.runs):
    print(i)
    direc_run=args.directory+"run"+str(i)+"/"
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)
        
    bkg = rv_bkg.rvs(args.N_bkg*2)
    sig = rv_sig.rvs(args.N_sig)
    print(bkg.shape, sig.shape)
    data = np.concatenate((bkg, sig))
    labels = to_categorical(np.append(np.zeros(args.N_bkg), np.ones(args.N_bkg+args.N_sig)))
    print(labels)

    inds = np.array(range(len(data)))
    np.random.shuffle(inds)
    inds_train, inds_test = np.array_split(inds, 2)
    X_train, X_test = data[inds_train], data[inds_test]
    Y_train, Y_test = labels[inds_train], labels[inds_test]

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

    preds = model.predict(X_test)
    np.save(direc_run+'classifier_history.npy', results.history)
    np.save(direc_run+"data_preds.npy", preds[Y_test[:,1]==1])
    np.save(direc_run+"BT_preds.npy", preds[Y_test[:,1]==0])
    model.save(direc_run+"model.keras")