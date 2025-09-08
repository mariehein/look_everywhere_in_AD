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
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--noearlystopping", default=False, action="store_true")
parser.add_argument("--uniform", default=False, action="store_true")
parser.add_argument("--ensemble", type=int, default=None)
parser.add_argument("--bins", type=int, default=255)


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
        data = oned_sample(40000, rv)
    else:
        data = rv.rvs(40000)
    X_train, X_test =np.array_split(data,2)

    Y_train = np.append(np.ones(len(X_train)//2), np.zeros(len(X_train)//2))
    np.random.shuffle(Y_train)

    model = HistGradientBoostingClassifier(max_bins=args.bins, early_stopping=not args.noearlystopping)
    model.fit(X_train, Y_train)

    data_test, BT_test = np.array_split(X_test, 2)
    np.save(direc_run+"test_data_preds.npy", model.predict_proba(data_test)[:,1])
    np.save(direc_run+"test_BT_preds.npy", model.predict_proba(BT_test)[:,1])
    np.save(direc_run+"train_data_preds.npy", model.predict_proba(X_train[Y_train==0])[:,1])
    np.save(direc_run+"train_BT_preds.npy", model.predict_proba(X_train[Y_train==1])[:,1])