#!/bin/bash
#SBATCH -J sss_known_cost
#SBATCH -o sss_known_cost_%j.out
#SBATCH -e sss_known_cost_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qx66@cornell.edu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --array=0-899
#SBATCH --mem-per-cpu=64G
#SBATCH -t 240:00:00
#SBATCH --partition=default_partition
#SBATCH --ntasks-per-node=1

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate automl_env

CONFIG_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" scripts/all_configs.txt)
echo "Running SLURM job $SLURM_ARRAY_TASK_ID"
echo "Config file: $CONFIG_FILE"
which python
python --version
ls "$CONFIG_FILE"
python scripts/sss_known_cost.py --config "$CONFIG_FILE"

conda deactivate