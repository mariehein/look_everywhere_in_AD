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
parser.add_argument("--dimensions", type=int, default = 2)
parser.add_argument("--bins_edge", type=float, default = 3)
args = parser.parse_args()

def to_categorical(Y, N_classes=2):
	Y=np.array(Y,dtype=int)
	return np.eye(N_classes)[Y]

rv = multivariate_normal(np.zeros(args.dimensions), np.diag(np.ones(args.dimensions)))
if not os.path.exists(args.directory):
	os.makedirs(args.directory)

class classifier():
    def __init__(self, bins, bins_edge, data, BT):
        self.edges = [np.linspace(-bins_edge, bins_edge, bins+1), np.linspace(-bins_edge,bins_edge,bins+1)]
        data_binned = np.histogramdd(data, bins=self.edges)[0]
        BT_binned = np.histogramdd(BT, bins=self.edges)[0]
        self.bins = bins
        with np.errstate(divide='ignore', invalid='ignore'):
            eps = 1e-20
            self.scores = (data_binned + eps) / len(data) / (BT_binned + eps) * len(BT)
        #self.scores = self.scores/(1+self.scores)
        #self.scores[(data_binned==0) & (BT_binned==0)] = 1
        #self.scores[(data_binned==0) & (BT_binned!=0)] = 0
        #self.scores[(BT_binned==0) & (data_binned!=0)] = 10000
        #print(sum((data_binned==0)), sum((BT_binned==0)))
        
    def find_bins(self, data):
        # Find the bin index for x and y
        x_bin = np.searchsorted(self.edges[0], data[:,0], side='right') - 1
        y_bin = np.searchsorted(self.edges[1], data[:,1], side='right') - 1

        x_bin[(x_bin<0) | (x_bin>=len(self.edges[0])-1)]
        x_bin[(y_bin<0) | (y_bin>=len(self.edges[1])-1)]

        return x_bin, y_bin

    def predict(self, data):
        locs = self.find_bins(data)
        locs[0][locs[0]==self.bins]=-1
        locs[1][locs[1]==self.bins]=-1
        scores = self.scores[locs[0], locs[1]]
        scores[locs[0]==-1]=-1
        scores[locs[1]==-1]=-1
        return scores

for i in range(args.start_runs, args.start_runs+args.runs):
    print(i)
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
        data_train = data[inds[0]]
        BT_train = BT[inds[0]]
        for j in range(1, args.folds-1):
            data_train = np.concatenate((data_train, data[inds[j]]))
            BT_train = np.concatenate((BT_train, BT[inds[j]]))
        data_test = data[inds[-1]]
        BT_test = BT[inds[-1]]

        model = classifier(args.bins, args.bins_edge, data_train, BT_train)

        data_preds[k] = model.predict(BT_test)        
        samples_preds[k] = model.predict(data_test)

    np.save(direc_run+"data_preds.npy", data_preds)
    np.save(direc_run+"BT_preds.npy", samples_preds)