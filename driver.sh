#!/bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --job-name=ludwig-bench
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --output=experiment-logs/min_exp_bash_out
#SBATCH --partition=jag-hi
#SBATCH --nodelist=jagupard10
#SBATCH --time=10-0

# activate your desired anaconda environment
#source activate py37-an_hf 
source activate py37-an-lb
# cd to working directory
cd .

# launch commands
python $@ # $@ to propagate all arguments
