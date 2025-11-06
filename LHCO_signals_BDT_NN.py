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
parser.add_argument("--evaluate_on_train", default=False, action="store_true", help="Evaluating on test data results in a 50-50 " \
"data split; to use the same code, statistics are increased for evaluation on training data")

args = parser.parse_args()

N_data = 120000
if args.evaluate_on_train:
    set_size = 120000
else: 
    set_size = 60000

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
data = np.concatenate((data_full[N_data:2*N_data-signal_real], signal[:signal_real]), axis=0)

# make arrays of predictions
BT_test_preds = np.zeros((args.runs, set_size))
data_test_preds = np.zeros((args.runs, set_size))
BT_train_preds = np.zeros((args.runs, set_size))
data_train_preds = np.zeros((args.runs, set_size))

start = time.time()
for i in tqdm.tqdm(range(args.runs)):    
    np.random.shuffle(data)
    np.random.shuffle(BT)    
    if args.evaluate_on_train: 
        BT_test = BT
        data_test = data
        BT_train = BT
        data_train = data   
    else:
        BT_train, BT_test = np.array_split(BT,2)
        data_train, data_test = np.array_split(data,2)

    X_train = np.concatenate((BT_train, data_train))
    Y_train =  np.append(np.zeros_like(BT_train[:,0]), np.ones_like(data_train[:,0]))

    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    # Train classifier and get preds
    preds_list = [BT_train, data_train, BT_test, data_test, X_test]
    if args.classifier=="BDT":
        BT_train_preds[i], data_train_preds[i],BT_test_preds[i], data_test_preds[i], roc_test_preds = cl.BDT_training_and_preds(args, X_train, Y_train, preds_list, i)
    else:
        BT_train_preds[i], data_train_preds[i],BT_test_preds[i], data_test_preds[i], roc_test_preds = cl.NN_training_and_preds(args, X_train, Y_train, preds_list, inputs=inputs)
    print("AUC: %.3f" % cl.plot_roc(roc_test_preds, Y_test,title="BDT",directory=args.directory+"Nsig_"+str(args.signal_number)+"/"))

print("Time taken: ", time.time()-start)
np.save(args.directory+"Nsig_"+str(args.signal_number)+"/test_data_preds.npy", data_test_preds)
np.save(args.directory+"Nsig_"+str(args.signal_number)+"/test_BT_preds.npy", BT_test_preds)
np.save(args.directory+"Nsig_"+str(args.signal_number)+"/train_data_preds.npy", data_train_preds)
np.save(args.directory+"Nsig_"+str(args.signal_number)+"/train_BT_preds.npy", BT_train_preds)