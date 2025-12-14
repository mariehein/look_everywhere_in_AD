import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline


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
max_SIC_lims=16

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

data_color = {"train": "black", "test": "red", "kfolds": "dodgerblue"}
data_name = {"train": "Evaulate on train", "test": "Evaluate on test", "kfolds": "Evaluate with k-folding"}
classifier_linestyle = {"BDT": "dashed", "NN": "solid", "NN_noearlystopping": "dotted"}
classifier_name = {"BDT": "BDT", "NN": "NN with early stopping", "NN_noearlystopping": "NN without early stopping"}
max_err = 0.2
points = np.array([])


def figure_start_calibration(N_tests):
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    counts = np.arange(1,N_tests+1)/N_tests
    ax = ax.reshape((1,3))
    for i in range(3):
        ax[0,i].plot(counts, counts, color="grey", linestyle="dashed")
        ax[0,i].set_xlabel("Empirical p values")
        ax[0,i].set_ylabel("Calculated p values")
        ax[0,i].set_xscale("log")
        ax[0,i].set_yscale("log")
    return fig, ax

def figure_end_calibration(fig, ax, name, N_tests, lines_to_remove=None):

    counts = np.arange(1,N_tests+1)/N_tests
    for i in range(3):
        xmin, xmax = ax[0,i].get_xlim()
        ymin, ymax = ax[0,i].get_ylim()
        ax[0,i].plot(counts, counts, color="grey", linestyle="dashed")
        #ax[0,i].set_xlim(min(counts),1)
        #ymin, ymax =ax[0,i].get_ylim()
        ax[0,i].set_xlim(xmin, xmax)
        ax[0,i].set_ylim(ymin, ymax)
    ax[0,1].legend(loc="lower right")
    if lines_to_remove is not None:
        for l in lines_to_remove:
            l.remove()
    fig.tight_layout(rect=[0.032, 0, 1, 1])
    plt.show()
    fig.savefig("plots/"+name+".pdf")

def plot_pvalues_single(ax, pvalues, counts, title, NNBDT=True):
    if NNBDT:
        for i in range(len(labels)):
            #med = np.median(pvalues[i], axis=0)
            np.arange(1, len(pvalues[i])+1)
            len(pvalues[i])
            curr_counts = np.arange(1, len(pvalues[i])+1)/len(pvalues[i])
            #ax[0].fill_between(curr_counts, np.percentile(pvalues[i], 16, axis=0), np.percentile(pvalues[i], 84, axis=0), color=colors[i], alpha=0.2)
            ax[0].plot(curr_counts, pvalues[i], colors[i], label=labels[i])
        ax[0].set_title(title)
    else: 
        ax[0].plot(counts, pvalues, "black")
        ax[0].set_title(title)

def plot_pvalues_errorband(ax, pvalues, counts, title, NNBDT=True, linestyle="solid", plot_perc=[True, True, True], label=True):
    if NNBDT:
        for i in range(len(labels)):
            med = np.median(pvalues[i], axis=0)
            curr_counts = np.arange(1, len(pvalues[i][0])+1)/len(pvalues[i][0])
            if plot_perc[i]:
                ax[0].fill_between(curr_counts, np.percentile(pvalues[i], 16, axis=0), np.percentile(pvalues[i], 84, axis=0), color=colors[i], alpha=0.2)
                label= labels[i] if label else None
                ax[0].plot(curr_counts, med, colors[i], label=label, linestyle=linestyle)
        ax[0].set_title(title)
    else: 
        ax[0].fill_between(counts, np.percentile(pvalues, 16, axis=0), np.percentile(pvalues, 84, axis=0), color="black", alpha=0.2)    
        ax[0].plot(counts, np.median(pvalues, axis=0), "black", linestyle=linestyle)
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
    fig.savefig("plots/calibration_curve_"+name+".pdf")

def prep_pvalues(ps, split_into, NNBDT):
    if NNBDT:
        p_fin = []
        for p in ps:
            np.random.shuffle(p)
            new_p = np.zeros((10, len(p)//10))
            p = p[:(len(p)//10)*10].reshape(10,len(p)//10)
            for j in range(10):
                new_p[j] = np.sort(p[j])
            p_fin.append(new_p)
        return p_fin
    else:
        np.random.shuffle(ps)
        ps = ps.reshape((split_into, len(ps)//split_into))
        return np.sort([np.sort(ps[i]) for i in range(10)])

def calibration_curve_plot_errorband(p_train, p_test, p_kfolds, N_tests, name, title=None, NNBDT=True, plot_perc=[True, True, False]):
    fig, ax = figure_start_calibration(N_tests)
    counts = np.arange(1,N_tests+1)/N_tests
    plot_pvalues_errorband(ax[:,0], p_train, counts, "Evaluate on train set", NNBDT=NNBDT, plot_perc=plot_perc)
    plot_pvalues_errorband(ax[:,1], p_test, counts, "Evaluate on test set", NNBDT=NNBDT, plot_perc=plot_perc)
    plot_pvalues_errorband(ax[:,2], p_kfolds, counts, "Evaluate with k-folding", NNBDT=NNBDT, plot_perc=plot_perc)
    if NNBDT:  
        ax[0,1].legend(loc="lower right")
    fig.text(0.02, 0.5, title, rotation=90, va='center', ha='center', fontsize=16)

    figure_end_calibration(fig, ax, "calibration_curve_"+name, N_tests)

def calibration_curve_plot_errorband_multiple(p_train, p_test, p_kfolds, N_tests, name, linestyle=["solid", "dashed"], names=["Without early stopping", "With early stopping"], title=None, NNBDT=True, plot_perc=[True,True,False]):
    fig, ax = figure_start_calibration(N_tests)
    counts = np.arange(1,N_tests+1)/N_tests
    for i in range(len(linestyle)):
        label = linestyle[i]=="solid"
        plot_pvalues_errorband(ax[:,0], p_train[i], counts, "Evaluate on train set", NNBDT=NNBDT, linestyle=linestyle[i], plot_perc=plot_perc, label=label)
        plot_pvalues_errorband(ax[:,1], p_test[i], counts, "Evaluate on test set", NNBDT=NNBDT, linestyle=linestyle[i], plot_perc=plot_perc, label=label)
        plot_pvalues_errorband(ax[:,2], p_kfolds[i], counts, "Evaluate with k-folding", NNBDT=NNBDT, linestyle=linestyle[i], plot_perc=plot_perc, label=label)
    if NNBDT:
        ax[0,1].legend(loc="lower right")
    fig.text(0.02, 0.5, title, rotation=90, va='center', ha='center', fontsize=16)
    lines_to_remove = []
    for l in range(len(linestyle)):
        l1, = ax[0,1].plot([], [], color="black", linestyle=linestyle[l], label=names[l])
        lines_to_remove.append(l1)
    figure_end_calibration(fig, ax, "calibration_curve_"+name, N_tests, lines_to_remove=lines_to_remove)

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
     
def plot_end_1D_multiple(ax, sig, ylim=max_SIC_lims, ylabel=None, ylims=None, title=None, bbox=None, loc="lower right", legend=True): 
    if legend:
        if bbox:
            ax.legend(loc=loc, bbox_to_anchor=bbox)
        else:
            ax.legend(loc=loc)
    if ylabel is None:
        ax.set_ylabel(r"$\max\ \epsilon_S/\sqrt{\epsilon_B}$")
    else: 
        ax.set_ylabel(ylabel)
    ax.set_xlabel(r"$N_{sig}$")
    ax.set_xticks(sig)
    ax.set_xlim(min(sig), max(sig))
    if ylims is None:
        ax.set_ylim(0, ylim)
    else:
        ax.set_ylim(*ylims)

    if title is not None:
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        a = 0.03
        ax.text(xmin + a * (xmax-xmin), ymin + (1-a) *(ymax-ymin) , title, size=plt.rcParams['axes.labelsize'], color='black', horizontalalignment='left', verticalalignment='top')


def plot_sic(ax, sic, sic_low, sic_upp, sig, color, label, normed=False, linestyle="solid", alpha=0.2, alpha_line=1.):
	ax.plot(sig, sic, color=color,label=label, linestyle=linestyle, marker='o', alpha=alpha_line)
	ax.fill_between(sig, sic_low, sic_upp, alpha=alpha, facecolor=color)

def sics_plotting(general_directory, signals_plot, signals, classifiers=["BDT", "NN", "NN_noearlystopping"], datausages=["train", "test", "kfolds"], signal_randomized=True):
    sic = np.zeros((len(signals),len(points)+2))
    sic_low = np.zeros((len(signals),len(points)+2))
    sic_upp = np.zeros((len(signals),len(points)+2))


    fig, ax = plt.subplots(1,len(classifiers), figsize=(15,5))
    for k,c in enumerate(classifiers): 
        legend = (c=="NN")
        rand = "randsignal/" if signal_randomized else ""
        for i, d in enumerate(datausages):
            for j,s in enumerate(signals):
                sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D("BDT", points, general_directory+"LHCO_"+c+"/"+rand+d+"/Nsig_"+str(s)+"/")
            plot_sic(ax[k], sic[:,-1], sic_low[:,-1], sic_upp[:,-1], signals, data_color[d], data_name[d])
        plot_end_1D_multiple(ax[k], signals_plot, title=classifier_name[c], loc = "upper left", bbox=(0.025, 0.9), legend=legend)
    fig.tight_layout()
    fig.savefig("plots/LHCO_signals.pdf")


class trials_factor():
    def __init__(self, pvalues, switch_to_linear=1e-2, s=5):
        self.splines = []
        self.lins = []
        self.min = np.min(pvalues)
        self.switch_to_linear = switch_to_linear
        self.switch_to_linear_log = np.log10(switch_to_linear)
        for i in range(len(labels)):
            curr_counts = np.arange(1, len(pvalues[i][0])+1)/len(pvalues[i][0])
            med = np.sort(np.median(np.log10(pvalues[i]), axis=0))
            self.splines.append(UnivariateSpline(med, np.log10(curr_counts), s=s))
            #spline_upper = UnivariateSpline(np.log10(curr_counts[start:]), np.percentile(np.log10(pvalues[i][:,start:]), 84, axis=0), s=s)
            #spline_lower = UnivariateSpline(np.log10(curr_counts[start:]), np.percentile(np.log10(pvalues[i][:,start:]), 16, axis=0), s=s)
            self.lins.append(LinearRegression())
            self.lins[i].fit(med[curr_counts<1e-2].reshape(-1, 1), np.log10(curr_counts[curr_counts<1e-2]))
        self.min_output = 1/len(pvalues[0][0])/10
    
    def predict(self, pvalue, i, plotting=False, return_both=False):
        pvalue = np.max([self.min*np.ones_like(pvalue)/10, pvalue], axis=0)
        lin_values = self.lins[i].predict(np.log10(pvalue.reshape(-1, 1)))
        spline_values = self.splines[i](np.log10(pvalue))
        mask = np.array(lin_values<self.switch_to_linear_log, dtype=int)
        if return_both:
            return 10**(mask*lin_values+(1-mask)*spline_values), lin_values, spline_values
        if plotting:
            return 10**(mask*lin_values+(1-mask)*spline_values)
        return np.max([10**(mask*lin_values+(1-mask)*spline_values), np.ones_like(pvalue)*self.min_output], axis=0)
    
    def plot(self, ax):
        x = np.logspace(np.log10(self.min), 0, 10000)
        for i in range(len(self.splines)):
            ax.plot(self.predict(x, i, plotting=True), x, color=colors[i], linestyle="dashed")
    
    def plot_both(self, ax):
        x = np.logspace(np.log10(self.min), 0, 10000)
        for i in range(len(self.splines)):
            y, lin, spl = self.predict(x, i, plotting=True, return_both=True)
            ax.plot(10**lin, x, color=colors[i], linestyle="dotted", label=labels[i]+" linear part")
            ax.plot(10**spl, x, color=colors[i], linestyle="dashdot", label=labels[i]+" spline part")
            ax.plot(y, x, color=colors[i], linestyle="dashed", label=labels[i]+" combined")

def plot_trials_factor(ax, p, tf, name, dont_plot_tf=False, plot_both=False):
    plot_pvalues_errorband(ax[:], p, None, name, NNBDT=True)
    xmin, xmax = ax[0].get_xlim()
    ymin, ymax = ax[0].get_ylim()
    if not dont_plot_tf:
        if plot_both:
            tf.plot_both(ax[0])
        else:   
            tf.plot(ax[0])
    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(ymin, ymax)

def plot_trials_factor_split(p_train, p_test, p_kfolds, name, title=None, ignore_train=False, plot_both=False):
    if ignore_train:
        tf_train = None
    else:
        tf_train = trials_factor(p_train, switch_to_linear=1e-2)
    tf_test = trials_factor(p_test, switch_to_linear=1e-2)
    tf_kfolds = trials_factor(p_kfolds, switch_to_linear=1e-2)

    N_tests = len(p_train[0][0])
    fig, ax = figure_start_calibration(N_tests)
    fig.text(0.02, 0.5, title, rotation=90, va='center', ha='center', fontsize=16)
    plot_trials_factor(ax[:,0], p_train, tf_train, "Evaluate on train set", dont_plot_tf=ignore_train, plot_both=plot_both)
    plot_trials_factor(ax[:,1], p_test, tf_test, "Evaluate on test set", plot_both=plot_both)
    plot_trials_factor(ax[:,2], p_kfolds, tf_kfolds, "Evaluate with k-folding", plot_both=plot_both)

    figure_end_calibration(fig, ax, "trials_factor_"+name, N_tests)

    return {"train": tf_train, "test": tf_test, "kfolds": tf_kfolds }

def signal_pvalues_corr_uncorr(p_corr, p_uncorr, signals, name, datausages=["train", "test", "kfolds"], title=None, ylim=1e-6, min_values=1e-5, legend=False):
    sigma3 = 1-0.9973
    sigma5 = 5.7*1e-7
    sigma1 = 1-0.6825

    fig, ax = plt.subplots(1,len(labels), figsize=(15,5))
    ax = ax.reshape(-1,1)

    k=0
    for i,l in enumerate(labels):
        for j,d in enumerate(datausages):
            ax[i,k].plot(signals, np.median(p_corr[d][:,:, i], axis=-1), color=data_color[d], label=data_name[d])
            min_values_bool = (np.median(p_corr[d][:,:,i], axis=-1) == min_values)
            ax[i,k].scatter(signals[min_values_bool], np.median(p_corr[d][:,:, i], axis=-1)[min_values_bool], color=data_color[d], marker="v")
            ax[i,k].scatter(signals[np.invert(min_values_bool)], np.median(p_corr[d][:,:, i], axis=-1)[np.invert(min_values_bool)], color=data_color[d], marker="o")
            ax[i,k].fill_between(signals, np.percentile(p_corr[d][:,:, i], 16, axis=-1), np.percentile(p_corr[d][:,:, i], 84, axis=-1), color=data_color[d], alpha=0.2)
            ax[i,k].plot(signals, np.median(p_uncorr[d][:,:, i], axis=-1), color=data_color[d], label=None, marker="o", linestyle="dashed")
        
        ax[i,k].set_yscale("log")
        ax[i,k].set_title(l)
        ax[i,k].axhline(sigma1, color="grey", linestyle="solid", label=r"1$\sigma$")
        ax[i,k].axhline(sigma3, color="grey", linestyle="dashed", label=r"3$\sigma$")
        ax[i,k].set_xlim(0,1000)
        ax[i,k].set_ylim(ylim,1)
        ax[i,k].set_xlabel(r"$N_{sig}$")
        ax[i,k].set_ylabel("Calculated p-values")
    if legend:
        ax[0,0].legend(loc="lower left")
    fig.text(0.02, 0.5, title, rotation=90, va='center', ha='center', fontsize=16)
    fig.tight_layout(rect=[0.032, 0, 1, 1])
    fig.savefig("plots/pvalues_signals_"+name+".pdf")