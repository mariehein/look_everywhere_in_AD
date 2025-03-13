import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 
import scipy.stats as stats
from sklearn.ensemble import HistGradientBoostingClassifier
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--start_runs", default=0, type=int)
parser.add_argument("--runs", default=1000, type=int)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--bins", type=int, default = 256)
args = parser.parse_args()

def to_categorical(Y, N_classes=2):
	Y=np.array(Y,dtype=int)
	return np.eye(N_classes)[Y]

rv = multivariate_normal([0,0], [[1,0],[0,1]])
if not os.path.exists(args.directory):
	os.makedirs(args.directory)

for i in range(args.start_runs, args.start_runs+args.runs):
	
    direc_run=args.directory+"run"+str(i)+"/"
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)
        
    data = rv.rvs(400000)
    data_train, data_test, BT_train, BT_test = np.array_split(data, 4)

    X_train = np.concatenate((data_train, BT_train), axis=0)
    Y_train = np.append(np.ones(len(data_train)), np.zeros(len(BT_train)))
    inds = np.arange(len(X_train))
    np.random.shuffle(inds)
    X_train = X_train[inds]
    Y_train = Y_train[inds]

    tree = HistGradientBoostingClassifier(validation_fraction=0.5, max_bins=args.bins, early_stopping=True)
    tree.fit(X_train, Y_train)

    np.save(direc_run+"data_preds.npy", tree.predict_proba(data_test))
    np.save(direc_run+"BT_preds.npy", tree.predict_proba(BT_test))