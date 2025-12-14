#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=toydata_BDT_NN
#SBATCH --account rwth0934

#SBATCH --output=../output/toy_BDT_NN_%A_%a.txt

#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
###SBATCH --array=16600,17800,18000,18200,18400,18600,18500,18700,19000,19100,19200,19300,19400,19500,19600,19800,19900,20000,20200,20400
###SBATCH --array=0-99900:100
#SBATCH --array=0-99900:100
###SBATCH --array=0,100,400,700,800,900,2600,2700,2800,2900,4400,4600,4700,4800,4900,5100,5400,5900,6000,6100,7100,7200,7300,7400,8800,9800,11300,11400,11500,11600,11900,12300,12400,12500,12600,12700,12800,13100,13900,14000,14100,15200,15300,15400,16000,16100,16200,18200,21900,22400,22500,22600,23700,24000,24400,24500,24600,25800,25900,26000,26100,26700,26800,26900,27000,27100,27800,27900,28000,28500,28600,28700,28800,28900,29200,29300,29400,29500,29600,29700,32100,32500,32600,32700,32800,32900,33000,33900,34000,34100,34200,34300,34400,34500,37800,38400,38500,38600,38700,38800,38900,39000,39100,41700,44700,46100,48200,48300,55100,56000,56100,60200,60300,60400,60500,64800,65200,66100,66200,66300,66400,66500,66600,66700,74000,82700,83800,83900,85100,85200,85300,85500,85600,85700,85800,85900,92100,92200,92300,92400,93800,93900,95100,95200,96200,96300,96400,97400,97500,98800,99200,99300,99400,99700

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