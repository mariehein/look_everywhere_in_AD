import numpy as np
import argparse
import statistics_utils as stats

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--N_runs", default=10000, type=int)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--save_directory", required=True, type=str)
parser.add_argument("--folds", default=5, type=int)
parser.add_argument("--len_arrays", default=100, type=int)
parser.add_argument("--kfolds", default=False, action="store_true")
parser.add_argument("--test", default=False, action="store_true")
args = parser.parse_args()

if args.kfolds:
    N_samples_after, N_samples, N_after, N = stats.make_arrays_kfolds(args.directory, runs=args.N_runs, folds=args.folds, len_arrays=args.len_arrays)
    p = stats.p_values_NN_BDT(N_samples_after, N_samples, N_after, N)
    np.save(args.save_directory+"kfolds.npy", p)
elif args.test:
    N_samples_after, N_samples, N_after, N = stats.make_arrays(args.directory, "train", runs=args.N_runs, len_arrays=args.len_arrays)
    p = stats.p_values_NN_BDT(N_samples_after, N_samples, N_after, N)
    np.save(args.save_directory+"evaluate_on_train_half.npy", p)

    N_samples_after, N_samples, N_after, N = stats.make_arrays(args.directory, "test", runs=args.N_runs, len_arrays=args.len_arrays)
    p = stats.p_values_NN_BDT(N_samples_after, N_samples, N_after, N)
    np.save(args.save_directory+"evaluate_on_test_half.npy", p)
else:
    N_samples_after, N_samples, N_after, N = stats.make_arrays(args.directory, "train", runs=args.N_runs, len_arrays=args.len_arrays)
    p = stats.p_values_NN_BDT(N_samples_after, N_samples, N_after, N)
    np.save(args.save_directory+"evaluate_on_train.npy", p)

    N_samples_after, N_samples, N_after, N = stats.make_arrays(args.directory, "test", runs=args.N_runs, len_arrays=args.len_arrays)
    p = stats.p_values_NN_BDT(N_samples_after, N_samples, N_after, N)
    np.save(args.save_directory+"evaluate_on_test.npy", p)
