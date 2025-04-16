import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 
import scipy.stats as stats
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--start_runs", default=0, type=int)
parser.add_argument("--runs", default=100, type=int)
parser.add_argument("--folds", default=5, type=int)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--bins", type=int, default = 255)
parser.add_argument('--dimensions', type=int, default =2)
parser.add_argument('--iterations', type=int, default =100)
parser.add_argument('--leaves', type=int, default =31)
args = parser.parse_args()

def to_categorical(Y, N_classes=2):
	Y=np.array(Y,dtype=int)
	return np.eye(N_classes)[Y]

rv = multivariate_normal(np.zeros(args.dimensions), np.diag(np.ones(args.dimensions)))
if not os.path.exists(args.directory):
	os.makedirs(args.directory)

for i in range(args.start_runs, args.start_runs+args.runs):

    direc_run=args.directory+"run"+str(i)+"/"
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)
        
    data = rv.rvs(250000)
    BT = rv.rvs(250000)
    data = np.array_split(data, 5)
    BT = np.array_split(BT, 5)

    data_preds = np.zeros((5,50000))
    samples_preds = np.zeros((5,50000))

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

        tree = HistGradientBoostingClassifier(validation_fraction=0.5, max_bins=args.bins, early_stopping=True, random_state=i, max_iter=args.iterations, max_leaf_nodes=args.leaves)
        tree.fit(X_train, Y_train)
        data_preds[k] = tree.predict_proba(BT_test)[:,1]
        samples_preds[k] = tree.predict_proba(data_test)[:,1]

    np.save(direc_run+"data_preds.npy", data_preds)
    np.save(direc_run+"BT_preds.npy", samples_preds)