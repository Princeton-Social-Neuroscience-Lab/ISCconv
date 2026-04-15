#!/bin/bash

# Run within pipeline code/ directory:
# sbatch slurm_denoising.sh

# Set partition
#SBATCH --partition=all

# How long is the job (HH:MM:SS)?
#SBATCH --time=00:10:00 

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem-per-cpu=12000

# Name of jobs?
#SBATCH --job-name=denoising

# Where to output log files?
#SBATCH --output='./code/logs/denoising-%A_%a.log'

# Update with your email 
#SBATCH --mail-user=####.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Number jobs to run in parallel, pass index subject ID 
#SBATCH --array=003,004,005,006,007,008,009,013,014,016,017,018,020,021,022,023,026,027,028,029,030,031,032,033,034,037,038,042,043,045,046,047,048,050,051,052,053,054,055,056,057,058,059,060,061,062,063,064,065,066,067,069,070,072,073,074,075,103,104,105,106,107,108,109,113,114,116,117,118,120,121,122,123,126,127,128,129,130,131,132,133,134,137,138,142,143,145,146,147,148,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,169,170,172,173,174,175

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID
date

# Set subject ID based on array index
printf -v subj "%03d" $SLURM_ARRAY_TASK_ID

# Run denoising on task-Black and task-Conv
echo "Running cleaning task-Black for sub-$subj"
python clean_black.py $subj

echo "Finished denoising task-Black for sub-$subj"
date
