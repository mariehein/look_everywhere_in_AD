#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=toydata_BDT_NN
#SBATCH --account rwth0934

#SBATCH --output=../output/toy_BDT_NN_%A_%a.txt

#SBATCH --time=0-3:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-10000:100

source /home/zu992399/.bashrc
conda activate cathode
 
cd ..

classifier=${1}
earlystopping=${2}

gen_direc="/hpcwork/zu992399/look_elsewhere_paper/"

# build base directory depending on earlystopping
if [[ "$earlystopping" == "True" ]]; then
    direc=${gen_direc}"toydata_${classifier}/"
else
    direc=${gen_direc}"toydata_${classifier}_noearlystopping/"
    extra_args="--noearlystopping"
fi

python toydata_BDT_NN.py --directory ${direc} --classifier ${1} --start_run ${SLURM_ARRAY_TASK_ID} --runs 100