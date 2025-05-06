#!/bin/bash
#SBATCH -J known_cost                # Job name
#SBATCH -o known_cost_%j.out         # Output file (%j expands to jobID)
#SBATCH -e known_cost_%j.err         # Error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=qx66@cornell.edu         # Email address to send results to
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --array=0-299                        # Number of jobs
#SBATCH --mem-per-cpu=32G                  # Server memory requested (per node)
#SBATCH -t 240:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition        # Request partition
#SBATCH --ntasks-per-node=1                  # Number of tasks per node

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate automl_env
wandb login
wandb agent 'ziv-scully-group/PandoraBayesOpt/wtmrqpzl' --count 1
conda deactivate
