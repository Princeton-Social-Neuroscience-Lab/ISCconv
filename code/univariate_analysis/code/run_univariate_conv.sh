#!/bin/bash
#SBATCH --job-name=univariate  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G         # memory per cpu-core (4G is default)
#SBATCH --time=5:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
python univariate_conv.py

