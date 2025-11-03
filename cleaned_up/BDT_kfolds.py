import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 
import scipy.stats as stats

from sklearn.ensemble import HistGradientBoostingClassifier
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--start_runs", default=0, type=int)
parser.add_argument("--runs", default=100, type=int)
parser.add_argument("--folds", default=5, type=int)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--noearlystopping", default=False, action="store_true")
parser.add_argument("--uniform", default=False, action="store_true")
parser.add_argument("--ensemble", type=int, default=None)
parser.add_argument("--bins", type=int, default=255)
parser.add_argument("--signal", type=int, default=None)


args = parser.parse_args()

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
	
for i in range(args.start_runs, args.start_runs+args.runs):
	direc_run=args.directory+"run"+str(i)+"/"
	if not os.path.exists(direc_run):
		os.makedirs(direc_run)

	if args.uniform:
		X = oned_sample(50000, rv)
	else:
		X = rv.rvs(50000)
	data, BT = np.array_split(X,2)
	if args.signal is not None: 
		rv = multivariate_normal([3.,3.], [[0.1,0],[0,.1]])
		data = np.concatenate((data[:-args.signal],rv.rvs(args.signal)), axis=0)
		np.random.shuffle(data)
	data = np.array_split(data, args.folds)
	BT = np.array_split(BT, args.folds)

	data_preds = np.zeros((5,5000))
	samples_preds = np.zeros((5,5000))

	for k in range(args.folds):
		inds = np.roll(np.array(range(5)), k)
		X_train = np.concatenate((data[inds[0]], BT[inds[0]]))
		Y_train = np.concatenate((np.ones(len(data[inds[0]])), np.zeros(len(BT[inds[0]]))))
		for j in range(1, args.folds-1):
			X_train = np.concatenate((X_train, data[inds[j]], BT[inds[j]]))
			Y_train = np.concatenate((Y_train, np.ones(len(data[inds[j]])), np.zeros(len(BT[inds[j]]))))
		data_test = data[inds[-1]]
		BT_test = BT[inds[-1]]

		inds = np.arange(len(X_train))
		np.random.shuffle(inds)
		X_train = X_train[inds]
		Y_train = Y_train[inds]

		for i in range(args.ensemble):
			model = HistGradientBoostingClassifier(max_bins=args.bins, early_stopping=not args.noearlystopping, validation_fraction=0.5)
			model.fit(X_train, Y_train)
			
			data_preds[k] += model.predict_proba(data_test)[:,1]/args.ensemble
			samples_preds[k] += model.predict_proba(BT_test)[:,1]/args.ensemble

	np.save(direc_run+"test_data_preds.npy", data_preds)
	np.save(direc_run+"test_BT_preds.npy", samples_preds)