#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=LHCO_BDT_NN
#SBATCH --account rwth0934

#SBATCH --output=../output/LHCO_BDT_NN_%A_%a.txt

#SBATCH --time=0-5:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
###SBATCH --array=0-10000:100
#SBATCH --array=8900,9500
source /home/zu992399/.bashrc
conda activate cathode
 
cd ..

classifier=${1}
earlystopping=${2}
data_usage=${3}
echo "Classifier: ${classifier}"
echo "Earlystopping: ${earlystopping}"
echo "Data usage: ${data_usage}"

gen_direc="/hpcwork/zu992399/look_elsewhere_paper/"

# build base directory depending on earlystopping
if [[ "$earlystopping" == "True" ]]; then
    direc=${gen_direc}"LHCO_${classifier}/"
else
    direc=${gen_direc}"LHCO_${classifier}_noearlystopping/"
    extra_args="--noearlystopping"
fi

if [[ "$data_usage" == "kfolds" ]]; then
    echo "kfolds"
    python LHCO_BDT_NN_kfolds.py --directory ${direc}"kfolds/" --classifier ${1} --start_run ${SLURM_ARRAY_TASK_ID} --runs 100 ${extra_args}
elif [[ "$data_usage" == "train" ]]; then
    echo "train"
    python LHCO_BDT_NN.py --directory ${direc}"train/" --classifier ${1} --start_run ${SLURM_ARRAY_TASK_ID} --runs 100 --evaluate_on_train ${extra_args}
else
    echo "test"
    python LHCO_BDT_NN.py --directory ${direc}"test/" --classifier ${1} --start_run ${SLURM_ARRAY_TASK_ID} --runs 100 ${extra_args}
fi