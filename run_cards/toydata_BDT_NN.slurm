#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=toydata_BDT_NN
#SBATCH --account rwth0934

#SBATCH --output=../output/toy_BDT_NN_%A_%a.txt

#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-99900:100

source /home/zu992399/.bashrc
conda activate cathode
 
cd ..

arr_len=100
classifier=${1}
earlystopping=${2}
kfolds=${3}
echo "Classifier: ${classifier}"
echo "Earlystopping: ${earlystopping}"
echo "Kfolding: ${kfolds}"

gen_direc="/hpcwork/zu992399/look_elsewhere_paper/"

# build base directory depending on earlystopping
if [[ "$earlystopping" == "True" ]]; then
    direc=${gen_direc}"toydata_${classifier}/"
else
    direc=${gen_direc}"toydata_${classifier}_noearlystopping/"
    extra_args="--noearlystopping"
fi

if [[ "$kfolds" == "True" ]]; then
    echo "kfolds"
    python toydata_BDT_NN_kfolds.py --directory ${direc}"kfolds/" --classifier ${1} --start_run ${SLURM_ARRAY_TASK_ID} --runs ${arr_len} ${extra_args}
else
    echo "traintest"
    python toydata_BDT_NN.py --directory ${direc}"traintest/" --classifier ${1} --start_run ${SLURM_ARRAY_TASK_ID} --runs ${arr_len} ${extra_args}
fi