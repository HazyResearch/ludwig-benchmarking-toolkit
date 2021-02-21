#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --job-name=ludwig-bench
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --output=experiment-logs/test_bohb_2
#SBATCH --partition=jag-hi
#SBATCH --nodelist=jagupard11
#SBATCH --time=10-0

# activate your desired anaconda environment
#source activate py37-an_hf 
source activate py37-an-lb
# cd to working directory
cd .

# launch commands
python $@ # $@ to propagate all arguments
