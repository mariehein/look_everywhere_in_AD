import numpy as np
import argparse
import statistics_utils as stats
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--N_runs", default=10000, type=int)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--save_directory", required=True, type=str)
parser.add_argument("--folds", default=5, type=int)
parser.add_argument("--len_arrays", default=100, type=int)
parser.add_argument("--kfolds", default=False, action="store_true")
args = parser.parse_args()

BH_percentiles = [1e-1, 1e-2, 1e-3, 1e-4]
fixed_cut = [0.51, 0.53, 0.55]

def calc_and_apply_threshold(samples_preds, data_preds, efficiency):
    """
    Returns number of samples and data events before and after cut

    Apply quantile cut based on efficiency to samples classifier scores and then the
    same threshold to data classifier scores 
    """
    eps = np.quantile(samples_preds, 1-efficiency, method="nearest")
    if efficiency == 1:
        eps=0.
    N_samples_after = np.size(np.where(samples_preds>eps))+1
    N_samples = len(samples_preds)
    N_after = np.size(np.where(data_preds>eps))
    N = len(data_preds)
    return N_samples_after, N_samples, N_after, N

def make_arrays_kfolds(folder, start_runs=0, runs=2100, len_arrays=100, folds=5):
    arr_shape = (folds, runs,len(BH_percentiles))
    N_samples_after = np.zeros(arr_shape)
    N_samples = np.zeros(arr_shape)
    N_after = np.zeros(arr_shape)
    N = np.zeros(arr_shape)

    for r in range(start_runs, runs):
        if r%len_arrays==0:
            samples_preds = np.load(folder+"runs/run"+str(r)+"_test_BT_preds.npy")
            data_preds = np.load(folder+"runs/run"+str(r)+"_test_data_preds.npy")
            i=0
        for fold in range(folds):
            for j, perc in enumerate(BH_percentiles):
                N_samples_after[fold, r,j], N_samples[fold, r,j], N_after[fold, r,j], N[fold, r,j] = calc_and_apply_threshold(samples_preds[i,fold], data_preds[i,fold], perc)
        i+=1
    
    return np.sum(N_samples_after, axis=0), np.sum(N_samples, axis=0), np.sum(N_after, axis=0), np.sum(N, axis=0)
    
def make_arrays(folder, name, start_runs=0, runs=10000, len_arrays=100):
    arr_shape = (runs,len(BH_percentiles))
    N_samples_after = np.zeros(arr_shape)
    N_samples = np.zeros(arr_shape)
    N_after = np.zeros(arr_shape)
    N = np.zeros(arr_shape)

    for r in range(start_runs, runs):
        if r%len_arrays==0:
            samples_preds = np.load(folder+"runs/run"+str(r)+"_"+name+"_BT_preds.npy")
            data_preds = np.load(folder+"runs/run"+str(r)+"_"+name+"_data_preds.npy")
            i=0
        for j, perc in enumerate(BH_percentiles):
            N_samples_after[r,j], N_samples[r,j], N_after[r,j], N[r,j] = calc_and_apply_threshold(samples_preds[i], data_preds[i], perc)
        i+=1
    if not os.path.exists(folder+name):
        os.makedirs(folder+name)
    return N_samples_after, N_samples, N_after, N

if args.kfolds:
    N_samples_after, N_samples, N_after, N = make_arrays_kfolds(args.directory, runs=args.N_runs, folds=args.folds, len_arrays=100)
    p = stats.NN_BDT_pvalues(N_samples_after, N_samples, N_after, N)
    np.save(args.save_directory+"kfolds.py", p)
else:
    N_samples_after, N_samples, N_after, N = make_arrays(args.directory, "train", runs=args.N_runs, len_arrays=args.len_arrays)
    p = stats.NN_BDT_pvalues(N_samples_after, N_samples, N_after, N)
    np.save(args.save_directory+"evaluate_on_train.py", p)
    make_arrays(args.directory, "test", runs=args.N_runs, len_arrays=args.len_arrays)
    p = stats.NN_BDT_pvalues(N_samples_after, N_samples, N_after, N)
    np.save(args.save_directory+"evaluate_on_test.py", p)
