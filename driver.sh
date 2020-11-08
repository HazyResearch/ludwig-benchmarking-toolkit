#!/bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --job-name=bert_nonhf_sanitycheck
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --output=/juice/scr/avanika/ludwig-benchmark-experiments/mini-text-out
#SBATCH --partition=jag-standard
#SBATCH --time=10-0

# activate your desired anaconda environment
#source activate py37-an_hf 
source activate py36-an-lb
# cd to working directory
cd .

# launch commands
python $@ # $@ to propagate all arguments
