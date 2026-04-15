#!/bin/bash
#SBATCH --job-name=fastsrm
#SBATCH --output=fastsrm_%j.log
#SBATCH --time=1:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 --mem-per-cpu=40000
#SBATCH --job-name=hyperalignmentSchaeferHOGlasser

# Where to output log files?
#SBATCH --output='logs/hyperalignmentSchaeferHOGlasser-%A_%a.log'
#SBATCH --output='logs/hyperalignmentSchaeferHOGlasser-%A_%a.log'

# Number jobs to run in parallel, pass index subject ID 
#SBATCH --array=003,004,005,006,007,008,009,013,014,016,017,018,020,021,022,023,026,027,028,029,030,031,032,033,034,037,038,042,043,045,046,047,048,050,051,052,053,054,055,056,057,058,059,060,061,062,063,064,065,066,067,069,070,072,073,074,075

PADDED_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
DYAD_ID=${PADDED_ID: -2}
echo "Starting hyperalignment and ISC for DYAD_ID=$DYAD_ID"

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge

# Run hyperalingment and ISC using Schaefer atlas:
python run_fastsrm_Schaefer.py \
    --dyad_id "$DYAD_ID" \
    --conditions generate read \
    --clean_dir ./data/derivatives/clean \
    --data_dir./data \
    --model model9 \
    --atlas_path ./data/atlases/Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz.npy \
    --atlas_label schaefer100p17n \
    --n_components 20 \
    --n_iter 10 \
    --save_dir /jukebox/tamir/lmt/conversation_pipeline/data \
    --do_isc


# Run hyperalingment and ISC for harvard oxford atlas:
python run_fastsrm_HO.py \
    --dyad_id "$DYAD_ID" \
    --conditions generate read \
    --clean_dir ./data/derivatives/clean \
    --data_dir ./data \
    --model model9_task \
    --atlas_path ./data/atlases/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz.npy \
    --atlas_label hosubcortical21p \
    --n_components 20 \
    --n_iter 10 \
    --save_dir ./data \
    --do_isc

# Run hyperalingment and ISC using Glasser atlas:
python run_fastsrm_Glasser.py \
    --dyad_id "$DYAD_ID" \
    --conditions generate read \
    --clean_dir ./data/derivatives/clean \
    --data_dir ./data \
    --model model9_task \
    --atlas_path ./data/atlases/glasser360MNI.nii.npy \
    --atlas_label glasser360p \
    --n_components 20 \
    --n_iter 10 \
    --save_dir ./data \
    --do_isc
