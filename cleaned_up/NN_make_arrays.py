import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--N_runs", default=10000, type=int)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--noearlystopping", default=False, action="store_true")
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
    N_samples_after = np.size(np.where(samples_preds>eps))
    N_samples = len(samples_preds)
    N_after = np.size(np.where(data_preds>eps))
    N = len(data_preds)
    return N_samples_after, N_samples, N_after, N

def apply_fixed_cut(samples_preds, data_preds, eps):
    """
    Returns number of samples and data events before and after cut

    Apply quantile cut based on efficiency to samples classifier scores and then the
    same threshold to data classifier scores 
    """
    N_samples_after = np.size(np.where(samples_preds>=eps))
    N_samples = len(samples_preds)
    N_after = np.size(np.where(data_preds>=eps))
    N = len(data_preds)
    return N_samples_after, N_samples, N_after, N

def make_arrays(folder, name, start_runs=0, runs=10000):
    arr_shape = (runs,len(BH_percentiles))
    N_samples_after = np.zeros(arr_shape)
    N_samples = np.zeros(arr_shape)
    N_after = np.zeros(arr_shape)
    N = np.zeros(arr_shape)

    for r in range(start_runs, runs):
        f = folder+"run"+str(r)+"/"
        samples_preds = np.load(f+name+"_BT_preds.npy")
        data_preds = np.load(f+name+"_data_preds.npy")
        for j, perc in enumerate(BH_percentiles):
            N_samples_after[r,j], N_samples[r,j], N_after[r,j], N[r,j] = calc_and_apply_threshold(samples_preds, data_preds, perc)
    if not os.path.exists(folder+name):
        os.makedirs(folder+name)
    np.save(folder+name+"/N_samples_after.npy", N_samples_after)
    np.save(folder+name+"/N_samples.npy", N_samples)
    np.save(folder+name+"/N_after.npy", N_after)
    np.save(folder+name+"/N.npy", N)

print("Evaluate test on train")
make_arrays(args.directory, "train", runs=args.N_runs)
make_arrays(args.directory, "test", runs=args.N_runs)
if not args.noearlystopping:
    make_arrays(args.directory, "splittrain", runs=args.N_runs)
    make_arrays(args.directory, "splitval", runs=args.N_runs)