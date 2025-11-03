import numpy as np
import NN_BDT_utils as cl
import statistics_utils as stats
import argparse
import time
import tqdm
import os
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--classifier", type=str, choices=["BDT", "NN"], required=True)

parser.add_argument("--start_runs", default=0, type=int)
parser.add_argument("--runs", default=100, type=int)
parser.add_argument("--ensemble", default=10, type=int)
parser.add_argument("--folds", default=5, type=int)
parser.add_argument("--noearlystopping", default=False, action="store_true")
parser.add_argument("--revert", default=True, action="store_false")

args = parser.parse_args()
N_data = 50000 # total set size
inputs = 2 # number of dimensions

if not os.path.exists(args.directory):
	os.makedirs(args.directory)
	os.makedirs(args.directory+"runs/")

# 2D standard normal distribution for background
rv = stats.multivariate_normal([0,0], [[1,0],[0,1]])	

# make arrays of predictions
data_preds = np.zeros((args.runs, 5,5000))
samples_preds = np.zeros((args.runs, 5,5000))

start = time.time()
for i in tqdm.tqdm(range(args.start_runs, args.start_runs+args.runs)):
    X = rv.rvs(N_data)
    data, BT = np.array_split(X, 2)
    data_folds = np.array_split(data, args.folds)
    BT_folds = np.array_split(BT, args.folds)

    fold_indices = np.arange(args.folds)
    for k in range(args.folds):
        test_fold = fold_indices[k]
        train_folds = np.delete(fold_indices, k)

        X_train = np.concatenate([np.concatenate((data_folds[t], BT_folds[t])) for t in train_folds])
        Y_train = np.concatenate([np.concatenate((np.ones(len(data_folds[t])), np.zeros(len(BT_folds[t])))) for t in train_folds])

        data_test = data_folds[test_fold]
        BT_test = BT_folds[test_fold]

        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        Y_train = Y_train[perm]

        # Train classifier and get preds
        preds_list = [BT_test, data_test]
        if args.classifier=="BDT":
            samples_preds[i-args.start_runs,k], data_preds[i-args.start_runs,k] = cl.BDT_training_and_preds(args, X_train, Y_train, preds_list, i)
        else:
            samples_preds[i-args.start_runs,k], data_preds[i-args.start_runs,k] = cl.NN_training_and_preds(args, X_train, Y_train, preds_list, inputs=inputs)


print("Time taken: ", time.time()-start)
np.save(args.directory+"runs/run"+str(args.start_runs)+"_test_data_preds.npy", data_preds)
np.save(args.directory+"runs/run"+str(args.start_runs)+"_test_BT_preds.npy", samples_preds)