import numpy as np
import scipy.stats as stats
from tqdm import tqdm


def calc_exp_onebin(rv, x_low, x_high, y_low, y_high):
    return rv.cdf([x_high, y_high])+rv.cdf([x_low, y_low])-rv.cdf([x_high, y_low])-rv.cdf([x_low, y_high])

def calc_probability(rv, edges_x, edges_y):
    exp = np.zeros((len(edges_x)-1,len(edges_y)-1))
    for i in range(len(edges_x)-1):
        for j in range(len(edges_y)-1):
            exp[i,j] = calc_exp_onebin(rv, edges_x[i], edges_x[i+1], edges_y[j], edges_y[j+1])
    return exp

def p_value_binomial(N_data, N_BT, N_total):
    return 1-stats.binom.cdf(N_data-1, N_total, N_BT/N_total)

def p_value_events(data, exp, bins_x, bins_y):
    hist, _ = np.histogramdd(data, [bins_x, bins_y])
    return p_value_binomial(hist, exp, len(data))


"""bin_edges = 7
N_bkg = 100000
N_tests = 10
signal_positions = np.arange(1, 6, 1)
signal_widths = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 1.])
N_bins = np.array([3, 5, 10, 25, 50])
N_sig = np.array([100, 200, 500, 1000, 2000, 5000])"""

bin_edges = 7
N_bkg = 1000000
N_tests = 10
signal_positions = np.arange(1, 3.1, 0.5)
signal_widths = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 1.])
N_bins = np.arange(5,76, 1)#np.array([3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50])
N_sig = np.array([316])#np.array([50,100,200])#, 200, 500, 1000, 2000, 5000])

background = stats.multivariate_normal([0,0], [[1,0], [0,1]])

edges = {}
bkg_prob = {}
bin_widths = {}
for bin in N_bins:
    edges[bin] = np.linspace(-bin_edges, bin_edges, bin+1)
    bin_widths[bin] = edges[bin][1]-edges[bin][0]
    #bkg_prob[bin] = calc_probability(background, edges[bin], edges[bin]) 

print(edges[5])

dimension_description = {"signal_positions": signal_positions, "signal_widths": signal_widths, "N_bins": N_bins, "N_sig": N_sig}
np.save("results/pvalues_binned/dimension_description.npy", dimension_description)
#signal_prob = np.zeros((len(signal_positions, len(signal_widths), len(N_bins), len(N_sig))))
expected_pvalue = np.zeros((len(signal_positions), len(signal_widths), len(N_bins), len(N_sig)))
expected_pvalue_corrpos = np.zeros((len(signal_positions), len(signal_widths), len(N_bins), len(N_sig)))
observed_pvalue_corrpos = np.zeros((len(signal_positions), len(signal_widths), len(N_bins), len(N_sig), N_tests))
for i, p in tqdm(enumerate(signal_positions)):
    for j, w in enumerate(signal_widths):
        print(w)
        sig = stats.multivariate_normal([p,p], [[w,0],[0,w]])
        for k, b in enumerate(N_bins):
            print(b)
            for l, s in enumerate(N_sig):
                bw = bin_widths[b]
                bkg_corrpos = N_bkg*calc_exp_onebin(background, p-bw, p+bw, p-bw, p+bw)
                sig_corrpos = s*calc_exp_onebin(sig, p-bw, p+bw, p-bw, p+bw)
                expected_pvalue_corrpos[i,j,k,l] = p_value_binomial(bkg_corrpos+sig_corrpos, bkg_corrpos, N_bkg+s)
                for m in range(N_tests):
                    bkg_test = background.rvs(N_bkg)
                    bkg_inds = np.histogram2d(bkg_test[:,0], bkg_test[:,1], bins=[[p-bw,p+bw],[p-bw, p+bw]])[0]
                    #bkg_inds = sum((p-bw<bkg_test[:,0]) & (p-bw<bkg_test[:,1]) & (p+bw>bkg_test[:,0]) & (p+bw>bkg_test[:,1]))
                    #bkg_test = bkg_test[bkg_inds]
                    signal_test = sig.rvs(s)
                    signal_inds = np.histogram2d(signal_test[:,0], signal_test[:,1], bins=[[p-bw,p+bw],[p-bw, p+bw]])[0]
                    #signal_inds = sum((p-bw<signal_test[:,0]) & (p-bw<signal_test[:,1]) & (p+bw>signal_test[:,0]) & (p+bw>signal_test[:,1]))
                    #signal_test = signal_test[signal_inds]
                    observed_pvalue_corrpos[i,j,k,l,m] = p_value_binomial(bkg_inds+signal_inds, bkg_corrpos, N_bkg+s)

np.save("results/pvalues_binned_exp/observed_pvalue_corrpos_to75_large.npy", observed_pvalue_corrpos)
np.save("results/pvalues_binned_exp/expected_pvalue_corrpos_to75_large.npy", expected_pvalue_corrpos)





