import numpy as np
import NN_BDT_utils as cl
import argparse
import time
import tqdm
import os
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory", required=True, type=str)
parser.add_argument("--classifier", type=str, choices=["BDT", "NN"], required=True)
parser.add_argument("--signal_number", default=1000, type=int)

parser.add_argument("--runs", default=10, type=int)
parser.add_argument("--ensemble", default=10, type=int)
parser.add_argument("--noearlystopping", default=False, action="store_true")
parser.add_argument("--revert", default=True, action="store_false")
parser.add_argument("--folds", default=5, type=int)

args = parser.parse_args()

N_data = 120000
inputs = 4 # number of dimensions

if not os.path.exists(args.directory):
	os.makedirs(args.directory)
if not os.path.exists(args.directory+"Nsig_"+str(args.signal_number)+"/"):
	os.makedirs(args.directory+"Nsig_"+str(args.signal_number)+"/")

# load 1.2M background events to pull sets from
data_full = np.load("data/bkg.npy")
signal = np.load("data/signal.npy")
X_test = np.load("data/X_test.npy")
Y_test = np.load("data/Y_test.npy")

#calulate fraction of signal events in SR and scale injected signal down
signal_real = int(args.signal_number*len(signal)/100000)

BT = data_full[:N_data]
data = np.concatenate((data_full[:N_data-signal_real], signal[:signal_real]), axis=0)

BT_preds = np.zeros((args.runs, args.folds, N_data//5))
data_preds = np.zeros((args.runs, args.folds, N_data//5))

start = time.time()
for i in tqdm.tqdm(range(args.runs)):
    np.random.shuffle(data)
    np.random.shuffle(BT)
    data_folds = np.array_split(data, args.folds)
    BT_folds = np.array_split(BT, args.folds)

    fold_indices = np.arange(args.folds)
    roc_test_preds = np.zeros((len(X_test)*5))
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
        preds_list = [BT_test, data_test, X_test]
        if args.classifier=="BDT":
            BT_preds[i,k], data_preds[i,k], roc_test_preds = cl.BDT_training_and_preds(args, X_train, Y_train, preds_list, i)
        else:
            BT_preds[i,k], data_preds[i,k], roc_test_preds = cl.NN_training_and_preds(args, X_train, Y_train, preds_list, inputs=inputs)
        print("AUC: %.3f" % cl.plot_roc(roc_test_preds, Y_test,title="BDT",directory=args.directory+"Nsig_"+str(args.signal_number)+"/"))

print("Time taken: ", time.time()-start)
np.save(args.directory+"Nsig_"+str(args.signal_number)+"/test_data_preds.npy", data_preds)
np.save(args.directory+"Nsig_"+str(args.signal_number)+"/test_BT_preds.npy", BT_preds)