import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings("ignore")

c_RWTH = {'b': '#00549F',  # blue
          'lb': '#8EBAE5', # light blue
          'dr': '#A11035', # dark red
          'g': '#57AB27',  # green
          't': '#006165',  # teal / petrol
          'o': '#F6A800',  # orange
          'lg': '#BDCD00', # light green
          'gr': '#646567', # gray
          'v': '#612158',  # violett
          'r': '#CC071E',  # red
          'tq': '#0098A1', # turquoise
          'p': '#7A6FAC'}  # purple

max_err=0.2

S = 20000
B = 312858

plt.rcParams['pgf.rcfonts'] = False
#plt.rcParams['font.serif'] = []
#plt.rcParams['font.family']="serif"
#pl.rcParams['mathtext.fontset']="cm"
plt.rcParams['figure.figsize'] = 7, 5
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['errorbar.capsize'] = 2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.frameon'] = True

labels = [r"$\epsilon_B = 10^{-1}$", r"$\epsilon_B = 10^{-2}$", r"$\epsilon_B = 10^{-3}$"]
colors = ["green", "blue", "red"]

def calibration_curve_plot(p_train, p_test, p_kfolds, N_tests, name, title=None, NNBDT=True):

    def plot_pvalues(ax, pvalues, counts, title, NNBDT=True):
        if NNBDT:
            for i in range(len(labels)):
                #med = np.median(pvalues[i], axis=0)
                curr_counts = np.arange(1, len(pvalues[i])+1)/len(pvalues[i])
                #ax[0].fill_between(curr_counts, np.percentile(pvalues[i], 16, axis=0), np.percentile(pvalues[i], 84, axis=0), color=colors[i], alpha=0.2)
                ax[0].plot(curr_counts, pvalues[i], colors[i], label=labels[i])
            ax[0].set_title(title)
        else: 
            ax[0].plot(counts, pvalues, "black")
            ax[0].set_title(title)

    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax = ax.reshape((1,3))
    counts = np.arange(1,N_tests+1)/N_tests
    for i in range(3):
        ax[0,i].set_xlabel("Empirical p values")
        ax[0,i].set_ylabel("Calculated p values")
        ax[0,i].set_xscale("log")
        ax[0,i].set_yscale("log")
        ax[0,i].plot(counts, counts, color="grey", linestyle="dashed")

        #if title is not None:
        #    ymin, ymax = ax[0,i].get_ylim()
        #    xmin, xmax = ax[0,i].get_xlim()
        #    a = 0.03
        #    ax[0,i].text(xmin + a * (xmax-xmin), ymin + (1-a) *(ymax-ymin) , title, size=plt.rcParams['axes.labelsize'], color='black', horizontalalignment='left', verticalalignment='top')
        
    counts = np.arange(1,N_tests+1)/N_tests
    plot_pvalues(ax[:,0], p_train, counts, "Evaluate on train set", NNBDT=NNBDT)
    plot_pvalues(ax[:,1], p_test, counts, "Evaluate on test set", NNBDT=NNBDT)
    plot_pvalues(ax[:,2], p_kfolds, counts, "Evaluate with k-folding", NNBDT=NNBDT) 
    if NNBDT:  
        ax[0,1].legend(loc="lower right")
    fig.tight_layout()
    plt.show()

