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
N_bkg = 100000
N_tests = 10
signal_positions = np.arange(1, 3.1, 0.5)
signal_widths = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 1.])
N_bins = np.arange(5,101, 1)#np.array([3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50])
N_sig = np.array([50,100,200])#, 200, 500, 1000, 2000, 5000])

background = stats.multivariate_normal([0,0], [[1,0], [0,1]])

edges = {}
bkg_prob = {}
bin_widths = {}
for bin in N_bins:
    edges[bin] = np.linspace(-bin_edges, bin_edges, bin+1)
    bin_widths[bin] = edges[bin][1]-edges[bin][0]
    bkg_prob[bin] = calc_probability(background, edges[bin], edges[bin]) 

print(edges[5])

dimension_description = {"signal_positions": signal_positions, "signal_widths": signal_widths, "N_bins": N_bins, "N_sig": N_sig}
np.save("results/pvalues_binned/dimension_description.npy", dimension_description)
#signal_prob = np.zeros((len(signal_positions, len(signal_widths), len(N_bins), len(N_sig))))
expected_pvalue = np.zeros((len(signal_positions), len(signal_widths), len(N_bins), len(N_sig)))
expected_pvalue_corrpos = np.zeros((len(signal_positions), len(signal_widths), len(N_bins), len(N_sig)))
observed_pvalue_likelihood_ratio = np.zeros((len(signal_positions), len(signal_widths), len(N_bins), len(N_sig), N_tests))
observed_pvalue_maxbin = np.zeros((len(signal_positions), len(signal_widths), len(N_bins), len(N_sig), N_tests))
chosen_bin_match = np.zeros((len(signal_positions), len(signal_widths), len(N_bins), len(N_sig), N_tests))
#signal_test = np.zeros((len(signal_positions, len(signal_widths), len(N_bins), len(N_sig), N_tests)))
#bkg_test = np.zeros((len(signal_positions, len(signal_widths), len(N_bins), len(N_sig), N_tests)))
for i, p in tqdm(enumerate(signal_positions)):
    for j, w in enumerate(signal_widths):
        print(w)
        sig = stats.multivariate_normal([p,p], [[w,0],[0,w]])
        for k, b in enumerate(N_bins):
            print(b)
            signal_prob = calc_probability(sig, edges[b], edges[b])
            for l, s in enumerate(N_sig):
                #print(p,w,b,s)
                pvals = p_value_binomial(bkg_prob[b]*N_bkg+signal_prob*s, bkg_prob[b]*N_bkg, N_bkg+s)
                #print(pvals)
                inds = np.unravel_index(np.nanargmin(pvals), pvals.shape)
                expected_pvalue[i,j,k,l] = pvals[inds]
                bw = bin_widths[b]
                bkg_corrpos = N_bkg*calc_exp_onebin(background, p-bw, p+bw, p-bw, p+bw)
                sig_corrpos = s*calc_exp_onebin(sig, p-bw, p+bw, p-bw, p+bw)
                #print(bw, bkg_corrpos, sig_corrpos)
                expected_pvalue_corrpos[i,j,k,l] = p_value_binomial(bkg_corrpos+sig_corrpos, bkg_corrpos, N_bkg+s)
                for m in range(N_tests):
                    bkg_test = background.rvs(N_bkg)
                    signal_test = sig.rvs(s)
                    p_values = p_value_events(np.concatenate((bkg_test, signal_test)),bkg_prob[b]*N_bkg, edges[b], edges[b])
                    observed_pvalue_likelihood_ratio[i,j,k,l,m] = p_values[inds]
                    inds2 = np.unravel_index(np.nanargmin(pvals), pvals.shape)
                    observed_pvalue_maxbin[i,j,k,l,m] = p_values[inds2]
                    chosen_bin_match[i,j,k,l,m] = int((inds2[0]==inds[0]) & (inds2[1]==inds[1]))

np.save("results/pvalues_binned/expected_pvalue.npy", expected_pvalue)
np.save("results/pvalues_binned/expected_pvalue_corrpos.npy", expected_pvalue_corrpos)
np.save("results/pvalues_binned/observed_pvalue_likelihood_ratio.npy", observed_pvalue_likelihood_ratio)
np.save("results/pvalues_binned/observed_pvalue_maxbin.npy", observed_pvalue_maxbin)
np.save("results/pvalues_binned/chosen_bin_match.npy", chosen_bin_match)





