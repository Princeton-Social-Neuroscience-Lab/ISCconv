"""Confound regression at run and trial level.

for model in phys_head_task phys_head_task_split phys phys_head ; do python code/clean.py -m $model; done
for model in model9 default ; do python code/clean.py -m $model; done
for model in model9_task default_task ; do python code/clean.py -m $model; done
for condition in read; do python code/clean.py -m $model -c read; done #LMT edit

LMT edited 02.07.2025: 
(1) added argument flag to parser for condition 
(2) flexible setting of conditions for run_level_regression & trial_level_regression functions
(3) file nomenclature changed from `fsaverage6` (since Zaid wrote this for .gii files) to 
    `MNI152NLin2009cAsym_desc-{condition}` so that information about the condition is now
    included in the file name for ease of use downstream. 
"""

import warnings

import os
import h5py
import numpy as np
import pandas as pd
from constants import (
    RUN_TRIAL_SLICE,
    RUN_TRS,
    RUNS,
    SUBS,
    CONVS,
    TR,
)
from nilearn import signal, masking
from nilearn.image import clean_img
from nilearn import image as nimg
from nilearn.glm.first_level import glover_hrf
import nibabel as nib

from scipy.stats import zscore
from tqdm import tqdm
from util import subject
from util.path import Path
from os.path import join as opj
from glob import glob



warnings.filterwarnings("ignore", category=DeprecationWarning)

HEAD_MOTION_CONFOUNDS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

# from Nastase et al., 2021
DEFAULT_CONFOUND_MODEL = {
    "confounds": HEAD_MOTION_CONFOUNDS + ["cosine"],
    "aCompCor": [{"n_comps": 5, "tissue": "CSF"}, {"n_comps": 5, "tissue": "WM"}],
}

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

CONFOUND_MODELS = {
    "model9": dict(confounds=CONFOUND_MODEL9),
    "default": dict(confounds=DEFAULT_CONFOUND_MODEL),
    "model9_task": dict(confounds=CONFOUND_MODEL9, add_task_confs=True),
    "default_task": dict(confounds=DEFAULT_CONFOUND_MODEL, add_task_confs=True),
    "default_task_trial": dict(confounds=DEFAULT_CONFOUND_MODEL, add_task_confs=True),
}

prep_dir = "./data/bids/derivatives/fmriprep" #originally in Lily's directory
mask_img_3d = nimg.load_img("./data/atlases/MNI152_T1_2mm_brain_mask.nii.gz")   

def check_dir(folder):
    if os.path.isdir(folder) == False:
        os.makedirs(folder)
        
def get_timinglog_run_regressors(sub_id: int, dft_run: pd.DataFrame):

    # trial boxcar
    trial_boxcar = np.zeros(RUN_TRS)
    for _, trial_slice in RUN_TRIAL_SLICE.items():
        trial_boxcar[trial_slice] = 1

    # prompt boxcar
    prompt_boxcar = np.zeros(RUN_TRS)
    for _, trial_slice in RUN_TRIAL_SLICE.items():
        prompt_slice = slice(trial_slice.start - 6, trial_slice.start)
        prompt_boxcar[prompt_slice] = 1

    screen_change = np.abs(np.diff(prompt_boxcar, prepend=0))

    speaker_role = "speaker" if sub_id > 100 else "listener"
    listener_role = "listener" if sub_id > 100 else "speaker"

    # create speaking and listening boxcars
    speech_onsets = (dft_run["run.time"] / TR).astype(int).to_numpy()
    button_press = np.zeros(RUN_TRS)
    receive_press = np.zeros(RUN_TRS)
    speech_boxcar = np.zeros(RUN_TRS)
    listen_boxcar = np.zeros(RUN_TRS)
    for i in range(len(speech_onsets) - 1):
        start = speech_onsets[i]
        stop = speech_onsets[i + 1]
        if dft_run.iloc[i]["role"] == speaker_role:
            speech_boxcar[start:stop] = 1
        elif dft_run.iloc[i]["role"] == listener_role:
            listen_boxcar[start:stop] = 1

        if (
            dft_run.iloc[i]["role"] == speaker_role
            and dft_run.iloc[i + 1]["role"] == listener_role
        ):
            button_press[stop] = 1
            screen_change[start] = 1
            screen_change[stop] = 1
        if (
            dft_run.iloc[i]["role"] == listener_role
            and dft_run.iloc[i + 1]["role"] == speaker_role
        ):
            receive_press[stop] = 1
            screen_change[start] = 1
            screen_change[stop] = 1

    return [
        trial_boxcar,
        prompt_boxcar,
        speech_boxcar,
        listen_boxcar,
        button_press,
        receive_press,
        screen_change,
    ]


def run_level_regression(model: str, condition:str, **kwargs):
    model_params = CONFOUND_MODELS[model]

    hrf = glover_hrf(TR, oversampling=1)

    if condition == "generate":
        cond_label = "G"
    elif condition == "read":
        cond_label = "R"
    else:
        # Fall back if you have other conditions
        cond_label = condition


    for sub_id in tqdm(SUBS):

        dft = subject.get_timing(sub_id, condition=cond_label)

        clean_bold = []
        run2trial = subject.get_trials(sub_id,cond_label)
        for run in RUNS:

            bold = subject.get_raw_bold(sub_id, runs=[run], trial_level=False)
            original_affine = bold.affine
            original_header = bold.header
            
            if np.shape(bold)[3] != 544:
                continue

            print(f"Input shape: {bold.shape}")

            confounds = subject.get_confounds(
                sub_id,
                runs=[run],
                trial_level=False,
                model_spec=model_params["confounds"],
            )
            dft_run = dft[dft.run == run]
            if model_params.get("add_task_confs", False):
                task_confounds = get_timinglog_run_regressors(sub_id, dft_run)
                task_confounds = [
                    np.convolve(confound, hrf, mode="full")[:RUN_TRS]
                    for confound in task_confounds
                ]
                task_confounds = np.stack(task_confounds).T
                confounds = np.hstack((confounds, task_confounds))
            print(confounds.shape)
                
            cleaned_bold = clean_img(
                bold,
                confounds=confounds,
                detrend=True,
                t_r=TR,
                ensure_finite=True,
                standardize="zscore_sample",
                standardize_confounds=True,
            )
            # slice out generate trials
            for trial in run2trial[run]:
                trial_slice = RUN_TRIAL_SLICE[trial]
                cleaned_bold_trial = cleaned_bold.slicer[:,:,:,trial_slice]
                cleaned_bold_trial = cleaned_bold_trial.get_fdata()  # shape (x, y, z, T)
                zscored_bold_trial = np.nan_to_num(zscore(cleaned_bold_trial, axis=-1))
                clean_bold.append(zscored_bold_trial)

        cleaned_bold = np.concatenate(clean_bold, axis=-1)
        print(f'Final shape of cleaned data: {cleaned_bold.shape}.')
        nii_img = nib.Nifti1Image(cleaned_bold, affine=original_affine, header=original_header) #<-----------LMT to do: figure out why loading this output is slow w/ nltools

        boldpath = Path(
            root="./data/derivatives/clean",
            datatype=model,
            sub=f"{sub_id:03d}",
            task="conv",
            space=f"MNI152NLin2009cAsym_desc-{condition}",
            ext=".nii.gz",
        )
        boldpath.mkdirs()
        nii_img.to_filename(boldpath)

        # with h5py.File(boldpath, "w") as f:
        #     f.create_dataset(name="bold", data=cleaned_bold)

def trial_level_regression(model: str, condition:str, **kwargs):
    model_params = CONFOUND_MODELS[model]

    hrf = glover_hrf(TR, oversampling=1)
    
    if condition == "generate":
        cond_label = "G"
    elif condition == "read":
        cond_label = "R"
    else:
        # Fallback if you have other conditions
        cond_label = condition

    for sub_id in tqdm(SUBS):
        dft = subject.get_timing(sub_id, condition=cond_label)

        clean_bold = []
        run2trial = subject.get_trials(sub_id, condition=cond_label)
        for run in RUNS:

            bold = subject.get_raw_bold(sub_id, runs=[run], trial_level=False)
            bold = bold.T

            confounds = subject.get_confounds(
                sub_id,
                runs=[run],
                trial_level=False,
                model_spec=model_params["confounds"],
            )
            dft_run = dft[dft.run == run]
            if model_params.get("add_task_confs", False):
                task_confounds = get_timinglog_run_regressors(sub_id, dft_run)
                task_confounds = [
                    np.convolve(confound, hrf, mode="full")[:RUN_TRS]
                    for confound in task_confounds
                ]
                task_confounds = np.stack(task_confounds).T
                confounds = np.hstack((confounds, task_confounds))

            # slice out generate trials
            for trial in run2trial[run]:
                trial_slice = RUN_TRIAL_SLICE[trial]

                cleaned_bold = signal.clean(
                    bold[trial_slice],
                    confounds=confounds[trial_slice],
                    detrend=True,
                    t_r=TR,
                    ensure_finite=True,
                    standardize="zscore_sample",
                    standardize_confounds=True,
                )

                clean_bold.append(cleaned_bold)

        cleaned_bold = np.vstack(clean_bold)

        boldpath = Path(
            root="./data/derivatives/clean",
            datatype=model,
            sub=f"{sub_id:03d}",
            task="conv",
            space=f"MNI152NLin2009cAsym_desc-{condition}",
            ext=".h5",
        )
        boldpath.mkdirs()
        with h5py.File(boldpath, "w") as f:
            f.create_dataset(name="data", data=cleaned_bold)


def main(model: str, **kwargs):
    # 1) Grab the 'condition' from kwargs (default "generate" if not given)
    condition = kwargs.get("condition", "generate")

    # 2) Specifying condition here
    if model.endswith("trial"):
        trial_level_regression(model, **kwargs)
    else:
        run_level_regression(model, **kwargs)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="default_task_trial")
    parser.add_argument("-v", "--verbose", action="store_true")
    #--------LMT adding here for condition flag
    parser.add_argument("-c", "--condition", type=str, default="generate",
                        help="Condition of interest, e.g. 'generate' or 'read'.")

    main(**vars(parser.parse_args()))
