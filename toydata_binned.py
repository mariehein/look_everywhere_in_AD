import numpy as np
import tqdm
import statistics_utils as stats

directory = "/hpcwork/zu992399/look_elsewhere/toydata_binned/"

bins = 5
bins_edge = 2
N = 10000
N_tests = 1000000
folds = 5

def perform_test_traintest(N_tests, bins, N, bins_edge = 2):
    rv = stats.multivariate_normal([0,0], [[1,0],[0,1]])
    edges = [np.linspace(-bins_edge,bins_edge,bins+1), np.linspace(-bins_edge,bins_edge,bins+1)]
    exp = stats.histogram_expectations(edges, bins, rv)

    p_train = np.zeros((bins,bins, N_tests))
    p_test = np.zeros((bins,bins, N_tests))
    for k in tqdm.tqdm(range(N_tests)):
        data = rv.rvs(size=N, random_state=k)
        hist_train = np.histogramdd(data, bins=edges)[0]
        p_train[:,:,k] = stats.p_value_binomial(hist_train, exp, N)

        data = rv.rvs(size=N, random_state=N_tests+k)
        hist = np.histogramdd(data, bins=edges)[0]
        p_test[:,:,k] = stats.p_value_binomial(hist, exp, N)

    min_ind_on_train = np.argmin(p_train.reshape((bins*bins, N_tests)), axis=0)
    p_train = np.array([p_train.reshape((bins*bins, N_tests))[min_ind_on_train,j] for j in range(N_tests)])
    p_test = np.array([p_test.reshape((bins*bins, N_tests))[min_ind_on_train,j] for j in range(N_tests)])
    
    return p_train, p_test 

def perform_test_kfolding(N_tests, bins, N, bins_edge = None, folds=5):
    rv = stats.multivariate_normal([0,0], [[1,0],[0,1]])
    edges = [np.linspace(-bins_edge,bins_edge,bins+1), np.linspace(-bins_edge,bins_edge,bins+1)]
    exp = stats.histogram_expectations(edges, bins, rv)

    p_add = np.zeros((N_tests, folds))
    p_train = np.zeros((N_tests, bins*bins, folds))
    
    for k in tqdm.tqdm(range(N_tests)):
        hist = np.zeros((bins,bins, folds))
        N_test = np.zeros((folds))
        exp_test = np.zeros((folds))
        best_bin = np.zeros((folds), dtype=int)

        data = np.array_split(rv.rvs(size=N, random_state=k), folds)
        for m in range(folds):
            hist[:,:,m] = np.histogramdd(data[m], bins=edges)[0]

        for m in range(folds):
            inds = np.roll(np.array(range(folds)), m)
            p_train[k,:,m] = stats.p_value_binomial(np.sum(hist[:,:,inds[:-1]],axis=-1), exp*(folds-1)/folds, N*(folds-1)/folds).flatten()

            best_bin[m] = np.argmin(p_train[k, :,m])
            N_test[m] = hist[:,:,inds[-1]].flatten()[best_bin[m]]
            exp_test[m] = exp.flatten()[best_bin[m]]/5

        p_add[k] = stats.p_value_binomial(np.sum(N_test), np.sum(exp_test), N)
    return p_add

p_train, p_test = perform_test_traintest(N_tests, bins, N, bins_edge)
np.save(directory+"p_train.npy", p_train)
np.save(directory+"p_test.npy", p_test)

p_kfolds = perform_test_kfolding(N_tests, bins, N, bins_edge, folds)
np.save(directory+"p_kfolds.npy", p_kfolds)
