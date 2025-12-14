#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=toydata_binned
###SBATCH --account rwth0934

#SBATCH --output=../output/toy_binned_%j.txt

#SBATCH --time=0-3:00:00
#SBATCH --ntasks=1
#SBATCH --mem=6G

source /home/zu992399/.bashrc
conda activate cathode
 
cd ..
python toydata_binned_R.py