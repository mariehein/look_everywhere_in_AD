import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--N_runs", default=10000, type=int)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--folds", default=5, type=int)
args = parser.parse_args()

BH_percentiles = [1e-1, 1e-2, 1e-3, 1e-4]
fixed_cut = [0.51, 0.53, 0.55]

BH_percentiles = [1e-1, 1e-2, 1e-3, 1e-4]
fixed_cut = [0.51, 0.53, 0.55]

def calc_and_apply_threshold(samples_preds, data_preds, efficiency):
    """
    Returns number of samples and data events before and after cut

    Apply quantile cut based on efficiency to samples classifier scores and then the
    same threshold to data classifier scores 
    """
    eps = np.quantile(samples_preds, 1-efficiency, method="nearest")
    #print(eps)
    if efficiency == 1:
        eps=0.
    N_samples_after = np.size(np.where(samples_preds>eps))+1
    N_samples = len(samples_preds)
    N_after = np.size(np.where(data_preds>eps))
    N = len(data_preds)
    #print(N_samples_after, N_samples, N_after, N)
    return N_samples_after, N_samples, N_after, N

def make_arrays(folder, start_runs=0, runs=2100, folds=5):
    arr_shape = (folds, runs,len(BH_percentiles))
    N_samples_after = np.zeros(arr_shape)
    N_samples = np.zeros(arr_shape)
    N_after = np.zeros(arr_shape)
    N = np.zeros(arr_shape)

    for r in range(start_runs, runs):
        f = folder+"run"+str(r)+"/"
        samples_preds = np.load(f+"test_BT_preds.npy")
        data_preds = np.load(f+"test_data_preds.npy")
        for fold in range(folds):
            for j, perc in enumerate(BH_percentiles):
                N_samples_after[fold, r,j], N_samples[fold, r,j], N_after[fold, r,j], N[fold, r,j] = calc_and_apply_threshold(samples_preds[fold], data_preds[fold], perc)
    
    np.save(folder+"N_samples_after.npy", np.sum(N_samples_after, axis=0))
    np.save(folder+"N_samples.npy", np.sum(N_samples, axis=0))
    np.save(folder+"N_after.npy", np.sum(N_after, axis=0))
    np.save(folder+"N.npy", np.sum(N, axis=0))

print("Evaluate test on train")
make_arrays(args.directory, runs=args.N_runs, folds=args.folds)