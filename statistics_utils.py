from scipy.stats import multivariate_normal 
import scipy.stats as stats
import numpy as np
import tqdm
import os

BH_percentiles = [1e-1, 1e-2, 1e-3, 1e-4]

def draw_toydata(N_data= 10000, mu=[0,0], sigma=[[1,0],[0,1]]):
    rv = multivariate_normal(mu, sigma)
    return  rv.rvs(mu, sigma, size=N_data), rv

def histogram_expectations(edges, bins, rv, N):
    exp = np.zeros((bins,bins))
    for i in range(bins):
        for j in range(bins):
            exp[i,j] = (rv.cdf([edges[0][i+1], edges[1][j+1]])+rv.cdf([edges[0][i], edges[1][j]]) - rv.cdf([edges[0][i+1], edges[1][j]])-rv.cdf([edges[0][i], edges[1][j+1]]))*N
    return exp

def p_value_binomial(N_data, N_BT, N_total, two_sided=True):
    p_value = 1-stats.binom.cdf(N_data-1, N_total, N_BT/N_total)
    if not two_sided:
        return p_value
    else:
        p_left = stats.binom.cdf(N_data, N_total, N_BT/N_total) 
        return np.min(np.array([p_value, p_left]), axis=0)*2

def p_value_poissonpoisson(N_data, N_BT, k=0.5, two_sided=True):
    p_value = 1-stats.nbinom.cdf(N_data-1, N_BT, k)
    if not two_sided:
        return p_value
    else:
        p_left = stats.nbinom.cdf(N_data, N_BT, k)
        return np.min(np.array([p_value, p_left]), axis=0)*2

def p_values_NN_BDT(N_samples_after, N_samples, N_after, N, two_sided=True):
    p = []
    for i in range(len(N[0])):
        p.append(np.sort(p_value_poissonpoisson(N_after[:,i], N_samples_after[:,i]/N_samples[:,i]*N[:,i], two_sided=two_sided)))
    return p

def p_values_from_folder(folder, datatype, two_sided=True):
    if datatype=="kfolds":
        loading_string=""
    elif datatype=="train":
        loading_string="train_"
    elif datatype=="test":
        loading_string="test_"
    N_samples_after = np.load(folder+loading_string+"N_samples_after.npy")
    N_samples = np.load(folder+loading_string+"N_samples.npy")
    N_after = np.load(folder+loading_string+"N_after.npy")
    N = np.load(folder+loading_string+"N.npy")
    return p_values_NN_BDT(N_samples_after, N_samples, N_after, N, two_sided=two_sided)

def calc_and_apply_threshold(samples_preds, data_preds, efficiency):
    """
    Returns number of samples and data events before and after cut

    Apply quantile cut based on efficiency to samples classifier scores and then the
    same threshold to data classifier scores 
    """
    eps = np.quantile(samples_preds, 1-efficiency, method="nearest")
    if efficiency == 1:
        eps=0.
    N_samples_after = np.size(np.where(samples_preds>=eps))
    N_samples = len(samples_preds)
    N_after = np.size(np.where(data_preds>=eps))
    N = len(data_preds)
    return N_samples_after, N_samples, N_after, N

def make_arrays_kfolds(folder, start_runs=0, runs=2100, len_arrays=100, folds=5):
    arr_shape = (folds, runs,len(BH_percentiles))
    N_samples_after = np.zeros(arr_shape)
    N_samples = np.zeros(arr_shape)
    N_after = np.zeros(arr_shape)
    N = np.zeros(arr_shape)

    for r in tqdm.tqdm(range(start_runs, runs)):
        if r%len_arrays==0:
            samples_preds = np.load(folder+"runs/run"+str(r)+"_test_BT_preds.npy")
            data_preds = np.load(folder+"runs/run"+str(r)+"_test_data_preds.npy")
            i=0
        for fold in range(folds):
            for j, perc in enumerate(BH_percentiles):
                N_samples_after[fold, r,j], N_samples[fold, r,j], N_after[fold, r,j], N[fold, r,j] = calc_and_apply_threshold(samples_preds[i,fold], data_preds[i,fold], perc)
        i+=1
    

    np.save(folder+"N_samples_after.npy", np.sum(N_samples_after, axis=0))
    np.save(folder+"N_samples_after.npy", np.sum(N_samples_after, axis=0))
    np.save(folder+"N_samples.npy", np.sum(N_samples, axis=0))
    np.save(folder+"N_after.npy", np.sum(N_after, axis=0))
    np.save(folder+"N.npy", np.sum(N, axis=0))

    return np.sum(N_samples_after, axis=0), np.sum(N_samples, axis=0), np.sum(N_after, axis=0), np.sum(N, axis=0)
    
def make_arrays(folder, name, start_runs=0, runs=10000, len_arrays=100):
    arr_shape = (runs,len(BH_percentiles))
    N_samples_after = np.zeros(arr_shape)
    N_samples = np.zeros(arr_shape)
    N_after = np.zeros(arr_shape)
    N = np.zeros(arr_shape)
    for r in tqdm.tqdm(range(start_runs, runs)):
        if r%len_arrays==0:
            samples_preds = np.load(folder+"runs/run"+str(r)+"_"+name+"_BT_preds.npy")
            data_preds = np.load(folder+"runs/run"+str(r)+"_"+name+"_data_preds.npy")
            i=0
        for j, perc in enumerate(BH_percentiles):
            N_samples_after[r,j], N_samples[r,j], N_after[r,j], N[r,j] = calc_and_apply_threshold(samples_preds[i], data_preds[i], perc)
        i+=1
    if not os.path.exists(folder+name):
        os.makedirs(folder+name)

    np.save(folder+name+"_N_samples_after.npy", N_samples_after)
    np.save(folder+name+"_N_samples.npy", N_samples)
    np.save(folder+name+"_N_after.npy", N_after)
    np.save(folder+name+"_N.npy", N)
    return N_samples_after, N_samples, N_after, N

def make_pvalues_traintest(filename, two_sided=True):
    BT = np.load(filename+"_BT_preds.npy")
    data = np.load(filename+"_data_preds.npy")

    arr_shape = (10, len(BH_percentiles))
    N_samples_after_train = np.zeros(arr_shape)
    N_samples_train = np.zeros(arr_shape)
    N_after_train = np.zeros(arr_shape)
    N_train = np.zeros(arr_shape)

    for i in range(10):
        for j, perc in enumerate(BH_percentiles):
            N_samples_after_train[i,j], N_samples_train[i,j], N_after_train[i,j], N_train[i,j] = calc_and_apply_threshold(BT[i], data[i], perc)
    return p_value_poissonpoisson(N_after_train, N_samples_after_train/N_samples_train*N_train, two_sided=two_sided)

def make_pvalues_kfolds(filename, folds=5, two_sided=True):
    BT = np.load(filename+"_BT_preds.npy")
    data = np.load(filename+"_data_preds.npy")
    arr_shape = (10, folds, len(BH_percentiles))
    N_samples_after_train = np.zeros(arr_shape)
    N_samples_train = np.zeros(arr_shape)
    N_after_train = np.zeros(arr_shape)
    N_train = np.zeros(arr_shape)

    for k in range(10):
        for i in range(folds):
            for j, perc in enumerate(BH_percentiles):
                N_samples_after_train[k,i,j], N_samples_train[k,i,j], N_after_train[k,i,j], N_train[k,i,j] = calc_and_apply_threshold(BT[k,i], data[k,i], perc)
    N_samples_after_train = np.sum(N_samples_after_train, axis=1)
    N_samples_train = np.sum(N_samples_train, axis=1)
    N_after_train = np.sum(N_after_train, axis=1)
    N_train = np.sum(N_train, axis=1)

    return p_value_poissonpoisson(N_after_train, N_samples_after_train/N_samples_train*N_train, two_sided=two_sided)