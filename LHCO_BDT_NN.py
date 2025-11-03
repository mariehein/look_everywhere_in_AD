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
parser.add_argument("--noearlystopping", default=False, action="store_true")
parser.add_argument("--revert", default=True, action="store_false")
parser.add_argument("--evaluate_on_train", default=False, action="store_true", help="Evaluating on test data results in a 50-50 " \
"data split; to use the same code, statistics are increased for evaluation on training data")

args = parser.parse_args()

if args.evaluate_on_train:
    N_data = 240000
else: 
    N_data = 120000

inputs = 4 # number of dimensions

if not os.path.exists(args.directory):
	os.makedirs(args.directory)
	os.makedirs(args.directory+"runs/")

# load 1.2M background events to pull sets from
data_full = np.load("data/bkg.npy")

# make arrays of predictions
BT_test_preds = np.zeros((args.runs, N_data//2))
data_test_preds = np.zeros((args.runs, N_data//2))
BT_train_preds = np.zeros((args.runs, N_data//2))
data_train_preds = np.zeros((args.runs, N_data//2))

start = time.time()
for i in tqdm.tqdm(range(args.start_runs, args.start_runs+args.runs)):
    idx = np.random.choice(len(data_full), 2 * N_data, replace=False)
    data = data_full[idx]

    X_train, X_test =np.array_split(data,2)
    Y_train = np.append(np.ones(len(X_train)//2), np.zeros(len(X_train)//2))
    np.random.shuffle(Y_train)
    BT_test, data_test = np.array_split(X_test,2)

    # Train classifier and get preds
    preds_list = [BT_test, data_test, X_train[Y_train==0], X_train[Y_train==1]]
    if args.classifier=="BDT":
        BT_test_preds[i-args.start_runs], data_test_preds[i-args.start_runs], BT_train_preds[i-args.start_runs], data_train_preds[i-args.start_runs] = cl.BDT_training_and_preds(args, X_train, Y_train, preds_list, i)
    else:
        BT_test_preds[i-args.start_runs], data_test_preds[i-args.start_runs], BT_train_preds[i-args.start_runs], data_train_preds[i-args.start_runs] = cl.NN_training_and_preds(args, X_train, Y_train, preds_list, inputs=inputs)


print("Time taken: ", time.time()-start)
np.save(args.directory+"runs/run"+str(args.start_runs)+"_test_data_preds.npy", data_test_preds)
np.save(args.directory+"runs/run"+str(args.start_runs)+"_test_BT_preds.npy", BT_test_preds)
np.save(args.directory+"runs/run"+str(args.start_runs)+"_train_data_preds.npy", data_train_preds)
np.save(args.directory+"runs/run"+str(args.start_runs)+"_train_BT_preds.npy", BT_train_preds)