from glob import glob
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from constants import RUN_TRIAL_SLICE, RUNS, TR
from util.extract_confounds import extract_confounds, load_confounds

from .path import Path
from nilearn.maskers import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask
from nilearn import image as nimg
from typing import List
from os.path import join as opj

def check_dir(folder):
    if os.path.isdir(folder) == False:
        os.makedirs(folder)

def get_conv(sub_id: int) -> str:
    return str(sub_id + 100 if sub_id < 100 else sub_id)


def get_partner(sub_id: int) -> int:
    return sub_id + 100 if sub_id < 100 else sub_id - 100


def recode_trial(trial: int) -> int:
    return ((int(trial) - 1) % 4) + 1


def get_timing(sub_id: int, condition: str) -> pd.DataFrame:
    timingpath = Path(
        root="./data/stimuli",
        conv=get_conv(sub_id),
        # datatype="timing",
        suffix="events",
        ext=".csv",
    )
    
    dft = pd.read_csv(timingpath)
    dft = dft[["run", "trial", "condition", "role", "comm.time", "run.time"]] 
        
    if condition is not None:
        dft = dft[dft.condition == condition]
        dft.dropna(subset=["comm.time"], inplace=True)

    return dft


def get_trials(sub_id: int, condition: str) -> dict:
    """Get which trials per run are a specific condition"""

    dft = get_timing(sub_id, condition=condition)
    dft2 = dft[["run", "trial", "condition"]].drop_duplicates().dropna()
    dft2["trial"] = dft2.trial.astype(int)
    rt_dict = dft2[["run", "trial"]].groupby("run")["trial"].apply(list).to_dict()
    return rt_dict


def get_transcript_switches(sub_id: int):
    """create a prodmask from transcript instead of timing log"""
    # prodmask impacts: clean.py when doing trial level, and encoding.

    df = get_transcript(subject=sub_id, modelname="model-gpt2-2b_layer-24")
    switches = []

    df2 = df[["run", "trial", "speaker", "start", "end"]].drop_duplicates()
    for _, group in df2.groupby(["run", "trial"]):
        group["turn"] = (group.speaker.diff() != 0).cumsum()
        group = group.groupby(["run", "trial", "turn"]).agg(
            dict(speaker="first", start="first", end="last")
        )
        group["start_tr"] = group.start / TR
        group["end_tr"] = group.end / TR

        trial_switches = np.zeros(120, dtype=int)
        for _, row in group[group.speaker == sub_id].iterrows():
            start = int(row.start_tr)
            end = int(row.end_tr)
            trial_switches[start:end] = 1

        switches.append(trial_switches)

    switches = np.concatenate(switches)
    return switches


def get_timinglog_boxcars(sub_id: int, condition: str):
    if condition == "generate":
        cond_label = "G"
    elif condition == "read":
        cond_label = "R"
    else:
        # Fallback if you have other conditions
        cond_label = condition    

    dft = get_timing(sub_id, condition=cond_label)

    role = "speaker" if sub_id > 100 else "listener"
    prod_boxcar = []
    for _, group in dft.groupby(["run", "trial"]):

        # switch onsets
        onsets = np.floor(group["comm.time"].values / TR).astype(int)
        trial_boxcar = np.zeros(120, dtype=int)
        s = 1 if group.iloc[0].role == role else 0
        for i in range(len(onsets) - 1):
            trial_boxcar[onsets[i] : onsets[i + 1]] = (i + s) % 2
        prod_boxcar.append(trial_boxcar)

    prod_boxcar = np.concatenate(prod_boxcar)
    button_idsA = np.clip(
        np.diff(prod_boxcar, prepend=prod_boxcar[0]), a_min=-1, a_max=0
    )
    button_idsB = np.clip(
        np.diff(prod_boxcar, prepend=prod_boxcar[0]), a_min=0, a_max=1
    )

    return prod_boxcar, button_idsA, button_idsB


def get_confounds(
    sub_id: int,
    model_spec: dict,
    runs: List[int] = RUNS,
    trial_level: bool = True,
):
    confound_path = Path(
        root="./data/bids/derivatives/fmriprep",
        sub=f"{sub_id:03d}",
        ses="1",
        datatype="func",
        task="Conv",
        run=0,
        desc="confounds",
        suffix="timeseries",
        ext=".tsv",
    )


    if trial_level:
        run2trial = get_trials(sub_id)

    all_confounds = []
    for run in runs:
        confound_path.update(run=run)
        print(confound_path)

        confounds_df, confounds_meta = load_confounds(confound_path.fpath)
        confounds_df.bfill(inplace=True)  # fill in nans when using derivatives
        confounds = extract_confounds(confounds_df, confounds_meta, model_spec)

        if trial_level:
            trials = run2trial[run]
            for trial in trials:
                trial_slice = RUN_TRIAL_SLICE[trial]
                all_confounds.append(confounds.iloc[trial_slice, :])
        else:
            all_confounds.append(confounds)

    all_confounds = np.vstack(all_confounds)

    return all_confounds

##########--------------LMT adaptation Zaid's code which was origininally on gifti to work on nifti files to denoise in volume space 

def get_raw_bold(
    sub_id: int,
    runs: List[int] = RUNS,
    trial_level: bool = True,
    run_time_mask=slice(None),
    voxel_mask=slice(None),
    concatenate: bool = True,
):
    # BIDS-like path
    bold_path = Path(
        root="./data/bids/derivatives/fmriprep",
        sub=f"{sub_id:03d}",
        ses="1",
        datatype="func",
        task="Conv",
        run=0,
        space="MNI152NLin2009cAsym_desc-preproc",
        suffix="bold",
        ext=".nii.gz",
    )

    # If trial-level, get run->trial dictionary
    if trial_level:
        run2trial = get_trials(sub_id)

    # Will store 2D arrays: each (n_voxels_in_mask, time)
    bold = []

    for run in runs:

        bold_path.update(run=run)
        print(bold_path)
        
        run_bold = nimg.load_img(bold_path)
        print(f" Shape of data: {run_bold.shape}--{sub_id}")

        # 3) If trial-level, slice each trial by time
        if trial_level:
            trials = run2trial.get(run, [])
            if not trials:
                print(f"No trials found for run {run}. Skipping or continuing.")
                continue

            for trial in trials:
                trial_slice = RUN_TRIAL_SLICE[trial]
                print(trial_slice)
                # Now (n_voxels, time_for_this_trial)
                trial_bold = run_bold.slicer[:,:,:,trial_slice]
                bold.append(trial_bold)

        else:
            # Keep entire run as one block
            bold.append(run_bold)

    # If nothing appended, return empty
    if not bold:
        return np.array([])

    # 4) Concatenate across runs/trials in time axis => (x,y,z,TR)
    final_bold = nimg.concat_imgs(bold)
    print(f" Shape after trimming trial: run-{run} {np.shape(final_bold)}--{sub_id}")

    return final_bold


def get_bold(
    sub_id: int,
    cache: str = None,
) -> np.ndarray:
    boldpath = Path(
        root="./data/derivatives/clean",
        datatype=cache,
        sub=f"{sub_id:03d}",
        task="conv",
        space= "MNI152NLin2009cAsym", #"fsaverage6",
        ext=".h5",
    )
    with h5py.File(boldpath, "r") as f:
        Y_bold = f["data"][...]

    return Y_bold
