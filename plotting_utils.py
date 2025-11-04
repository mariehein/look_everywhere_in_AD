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

def figure_start_calibration(N_tests):
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax = ax.reshape((1,3))
    counts = np.arange(1,N_tests+1)/N_tests
    for i in range(3):
        ax[0,i].set_xlabel("Empirical p values")
        ax[0,i].set_ylabel("Calculated p values")
        ax[0,i].set_xscale("log")
        ax[0,i].set_yscale("log")
        ax[0,i].plot(counts, counts, color="grey", linestyle="dashed")
    return fig, ax

def plot_pvalues_single(ax, pvalues, counts, title, NNBDT=True):
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

def plot_pvalues_errorband(ax, pvalues, counts, title, NNBDT=True):
    if NNBDT:
        for i in range(len(labels)):
            med = np.median(pvalues[i], axis=0)
            curr_counts = np.arange(1, len(pvalues[i])+1)/len(pvalues[i])
            ax[0].fill_between(curr_counts, np.percentile(pvalues[i], 16, axis=0), np.percentile(pvalues[i], 84, axis=0), color=colors[i], alpha=0.2)
            ax[0].plot(curr_counts, med, colors[i], label=labels[i])
        ax[0].set_title(title)
    else: 
        ax[0].fill_between(counts, np.percentile(pvalues, 16, axis=0), np.percentile(pvalues, 84, axis=0), color="black", alpha=0.2)    
        ax[0].plot(counts, np.median(pvalues, axis=0), "black")
        ax[0].set_title(title)

def calibration_curve_plot(p_train, p_test, p_kfolds, N_tests, name, title=None, NNBDT=True):
    fig, ax = figure_start_calibration(N_tests)
    counts = np.arange(1,N_tests+1)/N_tests
    plot_pvalues_single(ax[:,0], p_train, counts, "Evaluate on train set", NNBDT=NNBDT)
    plot_pvalues_single(ax[:,1], p_test, counts, "Evaluate on test set", NNBDT=NNBDT)
    plot_pvalues_single(ax[:,2], p_kfolds, counts, "Evaluate with k-folding", NNBDT=NNBDT) 
    if NNBDT:  
        ax[0,1].legend(loc="lower right")
    fig.tight_layout()
    plt.show()

def prep_pvalues(p, split_into, NNBDT):
    if NNBDT:
        return
    else:
        np.random.shuffle(p)
        p = p.reshape((split_into, len(p)//split_into))
        return np.sort([np.sort(p[i]) for i in range(10)])

def calibration_curve_plot_errorband(p_train, p_test, p_kfolds, N_tests, name, split_into=10, title=None, NNBDT=True):
    p_train = prep_pvalues(p_train, split_into, NNBDT)
    p_test = prep_pvalues(p_test, split_into, NNBDT)
    p_kfolds = prep_pvalues(p_kfolds, split_into, NNBDT)

    fig, ax = figure_start_calibration(N_tests)
    counts = np.arange(1,N_tests+1)/N_tests
    plot_pvalues_errorband(ax[:,0], p_train, counts, "Evaluate on train set", NNBDT=NNBDT)
    plot_pvalues_errorband(ax[:,1], p_test, counts, "Evaluate on test set", NNBDT=NNBDT)
    plot_pvalues_errorband(ax[:,2], p_kfolds, counts, "Evaluate with k-folding", NNBDT=NNBDT) 
    if NNBDT:  
        ax[0,1].legend(loc="lower right")
    fig.tight_layout()
    plt.show()

def read_ROC_SIC_1D(path, points, folder, N_runs=10, start_runs=0):
	fpr=np.load(folder+"fpr_"+path+".npy")[start_runs:start_runs+N_runs]
	#print(fpr.shape)
	tpr=np.load(folder+"tpr_"+path+".npy")[start_runs:start_runs+N_runs]
	ROC_values = np.zeros((len(fpr),len(points)))
	SIC_values = np.zeros((len(fpr),len(points)+2))
	for j in range(len(ROC_values)):
		inds = np.nonzero(tpr[j])[0]
		t = tpr[j, inds]
		f = fpr[j, inds]
		ROC_values[j] = interp1d(f, 1/f)(points)
		SIC_values[j,:-2] = interp1d(f, t/np.sqrt(f))(points)
		SIC_values[j,-1] = np.nanmax(np.nan_to_num(t/np.sqrt(f), posinf=0),where= f>1/312858/max_err**2,initial=0)
		SIC_values[j,-2] = np.nanmax(np.nan_to_num(t/np.sqrt(f),posinf=0))
	return np.median(SIC_values,axis=0), np.percentile(SIC_values, 16, axis=0), np.percentile(SIC_values, 84, axis=0)

def plot_end_1D(sig, name, small=True): 

	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
	plt.ylabel(r"$\max\ \epsilon_S/\sqrt{\epsilon_B}$")
	plt.xlabel(r"$N_{sig}$")
	plt.xticks(sig)
	plt.savefig("plots/1D_"+name+".pdf")
	plt.show()

def plot_sic(sic, sic_low, sic_upp, sig, color, label, normed=False, linestyle="solid"):
	if normed:
		n = np.max(sic)
		plt.plot(sig, sic/n, color=color,label=label, linestyle=linestyle)
		plt.fill_between(sig, sic_low/n, sic_upp/n, alpha=0.2, facecolor=color)
		return
	plt.plot(sig, sic, color=color,label=label, linestyle=linestyle, marker="o")
	plt.fill_between(sig, sic_low, sic_upp, alpha=0.2, facecolor=color)