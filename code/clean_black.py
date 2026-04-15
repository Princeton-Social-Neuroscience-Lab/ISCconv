"""Confound regression at run level for task Black.
This script will only run the basic model9 confound modeling 
approach. 
"""

import warnings

import os
from os.path import splitext
import time
import h5py
import json
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import signal, image
from nilearn.glm.first_level import glover_hrf
from scipy.stats import zscore
from tqdm import tqdm
from glob import glob
from scipy.stats import zscore
from os.path import join as opj
from nltools.data import Brain_Data

warnings.filterwarnings("ignore", category=DeprecationWarning)


def check_dir(folder):
    if os.path.isdir(folder) == False:
        os.makedirs(folder)
# Function for extracting group of (variable number) confounds
def extract_group(confounds_df, groups):

    # Expect list, so change if string
    if isinstance(groups, str):
        groups = [groups]

    # Filter for all columns with label
    confounds_group = []
    for group in groups:
        group_cols = [col for col in confounds_df.columns if group in col]
        confounds_group.append(confounds_df[group_cols])
    confounds_group = pd.concat(confounds_group, axis=1)

    return confounds_group

# Function for loading in confounds files
def load_confounds(confounds_fn):

    # Load the confounds TSV files
    confounds_df = pd.read_csv(confounds_fn, sep="\t")

    # Load the JSON sidecar metadata
    with open(splitext(confounds_fn)[0] + ".json") as f:
        confounds_meta = json.load(f)

    return confounds_df, confounds_meta


# Function for extracting confounds (including CompCor)
def extract_confounds(confounds_df, confounds_meta, model_spec):

    # Pop out confound groups of variable number
    groups = set(model_spec["confounds"]).intersection(["cosine", "motion_outlier"])

    # Grab the requested confounds
    confounds = confounds_df[[c for c in model_spec["confounds"] if c not in groups]]

    # Grab confound groups if present
    if groups:
        confounds_group = extract_group(confounds_df, groups)
        confounds = pd.concat([confounds, confounds_group], axis=1)

    # Get aCompCor / tCompCor confounds if requested
    compcors = set(model_spec).intersection(["aCompCor", "tCompCor"])
    if compcors:
        for compcor in compcors:
            if isinstance(model_spec[compcor], dict):
                model_spec[compcor] = [model_spec[compcor]]

            for compcor_kws in model_spec[compcor]:
                confounds_compcor = extract_compcor(
                    confounds_df, confounds_meta, method=compcor, **compcor_kws
                )

                confounds = pd.concat([confounds, confounds_compcor], axis=1)

    return confounds

        
def get_raw_bold(
    sub_id: int,
    run_time_mask=slice(None),
    voxel_mask=slice(None),
    concatenate: bool = True,
):
    bold_path = glob(opj(prep_dir, sub_id, "ses-1", "func",f"{sub_id}*Black*run-1*MNI*preproc_bold*gz"))[0]
    
    # Will store 2D arrays using nltools because masking to MNI is built in one step!
    img = Brain_Data(bold_path)
    run_bold = img.data.T

#     # Will store 2D arrays: each (n_voxels_in_mask, time)
#     img = nib.load(bold_path)
#     run_bold = img.get_fdata()  # shape => (X, Y, Z, T)

#     # 1) Flatten to 2D: (n_voxels, T)
#     nx, ny, nz, nt = run_bold.shape
#     run_bold = run_bold.reshape(-1, nt)

    # # 2) Apply voxel mask (first dimension) and time mask (second dimension)
    # run_bold = run_bold[voxel_mask, :][:, run_time_mask]
    print(f" Shape of data after flattening/masking: {run_bold.shape}--{sub_id}")
    # Keep entire run as one block

    return run_bold

def denoising_black(sub_id):
    """
    Denoise "I knew I was black" listening task with nilearn instead of nltools.
    For each subject, this function:

    1) Loads MNI preprocessed data
    2) Smooths with FWHM
    3) Reads and filters confounds
    4) Regresses out confounds with nilearn.signal.clean()
    5) Saves the residual (cleaned) 4D data as HDF5

    Parameters
    ----------
    sub_id : str
        Subject ID.
    """
    start = time.time()
    print(f"\nStarting Subject: {sub_id}")
    
    # ----------------------------------------------------------------------
    # 1) Get timeseries data
    # ----------------------------------------------------------------------
    hrf = glover_hrf(TR, oversampling=1)
    runs = 1
    bold = get_raw_bold(sub_id)
    bold = bold.T

    # ----------------------------------------------------------------------
    # 2) Read confounds tsv and filter columns
    # ----------------------------------------------------------------------
    conf_tsv_glob = glob(opj(prep_dir, sub_id, "ses-1", "func",
                                 f"{sub_id}*Black*run-1*confounds_timeseries.tsv"))[0]
    conf_files = glob(conf_tsv_glob)
    print(conf_tsv_glob)
    if not conf_files:
        print(f"No confounds file found for {sub_id}, skipping...")
        return
    conf_tsv = conf_files[0]
    confounds_df, confounds_meta  = load_confounds(conf_tsv)
    confounds_df.bfill(inplace=True)  # fill in nans when using derivatives
    confounds = extract_confounds(confounds_df, confounds_meta, CONFOUND_MODEL9)

    # ----------------------------------------------------------------------
    # 4) Regress out confounds with nilearn.signal.clean
    #    If you already added an intercept to mc_np, set detrend=False.
    #    If you prefer letting signal.clean handle linear detrending,
    #    skip the intercept column and set detrend=True.
    # ----------------------------------------------------------------------
    cleaned_bold = signal.clean(
        bold,
        confounds=confounds,       # shape (time, n_confounds)
        t_r=TR,
        detrend=True,         # we explicitly have an intercept now
        standardize="zscore",     
        ensure_finite=True,
        standardize_confounds=True  
    )

    # Slice out first 8 and last 8 from the time dimension
    n_timepoints = cleaned_bold.shape[1]
    drop_front = 8
    drop_back = 8
    cleaned_bold_trim = cleaned_bold[drop_front:-drop_back,:]
    zscored_cleaned_bold_trim = np.nan_to_num(zscore(cleaned_bold_trim)) # do I need this??
    print(f'Final shape of cleaned data: {np.shape(zscored_cleaned_bold_trim)}.')
    
    # To make it more compatible with the remaining pipeline, I'm saving as a brain instance using nltools
    denoised_data = Brain_Data()
    denoised_data.data = zscored_cleaned_bold_trim
    
    # ----------------------------------------------------------------------
    # 5) Save final residual data as .hdf5 (like your nltools code)
    # ----------------------------------------------------------------------
    write_dir = opj(out_dir, f"{sub_id}/model9")
    check_dir(write_dir)
    boldpath = opj(write_dir,f'{sub_id}_task-black_space-MNI152NLin2009cAsym.h5')
    # with h5py.File(boldpath, "w") as f:
    #     f.create_dataset(name="bold", data=zscored_cleaned_bold_trim)
    denoised_data.write(boldpath)
# ----------------------------------------------------------------------
# Set Variables 
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run confound modeling for task-Black")
    parser.add_argument("subj", type=str, help="Subject ID")
    args = parser.parse_args()

    sub_id = "sub-" + args.subj
    prep_dir = "./data/bids/derivatives/fmriprep"
    out_dir = "./data/derivatives/clean"
    task = "black"
    TR = 1.5
    # from Speer et al., 2023
    CONFOUND_MODEL9 = {
        "confounds": [
        "cosine",
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        # squared
        "trans_x_power2",
        "trans_y_power2",
        "trans_z_power2",
        "rot_x_power2",
        "rot_y_power2",
        "rot_z_power2",
        # derivatives
        "trans_x_derivative1",
        "trans_y_derivative1",
        "trans_z_derivative1",
        "rot_x_derivative1",
        "rot_y_derivative1",
        "rot_z_derivative1",
        # derivative powers
        "trans_x_derivative1_power2",
        "trans_y_derivative1_power2",
        "trans_z_derivative1_power2",
        "rot_x_derivative1_power2",
        "rot_y_derivative1_power2",
        "rot_z_derivative1_power2",
        "white_matter",
        "white_matter_power2",
        "white_matter_derivative1",
        "white_matter_derivative1_power2",
        "csf",
        "csf_power2",
        "csf_derivative1",
        "csf_derivative1_power2",
        ]
    }
    denoising_black(sub_id)
