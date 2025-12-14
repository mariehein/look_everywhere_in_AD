# Look Everywhere Effects in Anomaly Detection

This repository contains the code for the following paper:

*Look Everywhere Effects in Anomaly Detection*
By Marie Hein, Ben Nachman and David Shih.

## Reproducing the paper results 
In order to make the reproduction of the paper plots easier, the run cards used to produce the paper results are available in "run_cards". The structure of the code and how the run cards can be used to produce the paper runs is explained below.

The LHCO data is obtained using ```LHCO_dataprep.ipynb``` but final files can be found in folder "data".

### Binned analysis

For the binned analysis, as this runs quickly, there is a single file, which produces all paper p-values. This is done in ```toydata_binned_R.py```. ```toydata_binned.py```is an alternative code version, which chooses the best bin based on the lowest p-value instead of the highest $R(x)$. The directory for this run is set in the python file as are all other parameters. 

### Toy data BDT and NN runs

For the BDT and NN, all runs are started using ```toydata_BDT_NN.slurm``` by passing three options: 

1. Pass "BDT" or "NN" to choose classifier.
2. Pass "True" or "False" for whether early stopping should be used ("True" meaning early stopping is active).
3. Pass "True" or "False for whether k-fold cross validation should be used ("True" meaning k-folding is active).

Depending on the k-folding option, either ```toydata_BDT_NN.py``` or ```toydata_BDT_NN_kfolds.py``` is started by submitting a slurm job array, which performs the 100000 runs each needed for the calibration plots. The other parameters are used to assign the save directory and pass arguments to the scripts. 

### Background-only LHCO BDT and NN runs

For the BDT and NN, all runs are started using ```LHCO_BDT_NN.slurm``` by passing three options: 

1. Pass "BDT" or "NN" to choose classifier.
2. Pass "True" or "False" for whether early stopping should be used ("True" meaning early stopping is active).
3. Pass "train", "test" or "kfolds" to specify how evaluations should be performed.

Depending on the k-folding option, either ```LHCO_BDT_NN.py``` or ```LHCO_BDT_NN_kfolds.py``` is started by submitting a slurm job array, which performs the 100000 runs each needed for the calibration plots. The other parameters are used to assign the save directory and pass arguments to the scripts. 

### Signal LHCO BDT and NN runs

For the BDT and NN, all runs are started using ```LHCO_signals_BDT_NN.slurm``` by passing three options: 

1. Pass "BDT" or "NN" to choose classifier.
2. Pass "True" or "False" for whether early stopping should be used ("True" meaning early stopping is active).
3. Pass "train", "test" or "kfolds" to specify how evaluations should be performed.
4. Pass "True" or "False for whether signal should be chosen randomly for each of the ten performed runs ("True" meaning randomization is active).

Depending on the k-folding option, either ```LHCO_signals_BDT_NN.py``` or ```LHCO_signals_BDT_NN_kfolds.py``` is started by submitting a slurm job array of different signal injections. The other parameters are used to assign the save directory and pass arguments to the scripts. 

### Obtaining p-values for BDT and NN runs

The BDT and NN runs save prediction arrays _not_ p-values. To obtain the p-values it is necessary to call ```NN_BDT_make_arrays.py``` (in the background-only cases. For signal runs, p-value calculation is performed directly in plotting notebook) before running plotting scripts. For all used $\epsilon_B$, this saves the number of BT events before and after cut and the number of SR events before and after cut, which are needed to calculate the p-values in the plotting scripts. 

### Helper files

1. ```statistics_utils.py``` contains all helpers for p-value calculations including placing cuts for the NN and BDT.
2. ```NN_BDT_utils.py``` contains everything related to the BDT and NN training. By passing arguments, training data and list of arrays, for which predictions need to be obtained, training is then performed as specified and returns a list of predictions.

### Changes needed to be made to run cards on different systems

Beside changes to different submission system, which may need to be made, most importantly the python environment activation and general save directory need to be adjusted to your system.


## Plotting paper results

If all runs are performed with the run cards as given here and only a changed general save directory, it should only be necessary to adjust ```general_directory``` at the top of both ```plotting_toydata.ipynb``` and ```plotting_LHCO``` and run the whole notebook for all paper plots to be created. All plotting functions are contained in ```plotting_utils.py``` to kepp plotting notebooks minimal.







