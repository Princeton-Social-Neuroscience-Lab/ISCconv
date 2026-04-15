# fMRI hyperscanning shows that real-time conversation aligns minds through neural alignment
This repository contains code and resources for the pregistered study that can be found here:
https://osf.io/vs27z/?view_only=a8b97b5824a2473aa3f1590d36657fc5

**Author:** Laetitia Mwilambwe-Tshilobo
**Last edited:** 02/06/2025\ **Created**1/27/2024

# The pipeline steps
All scripts used to perform analyses from the paper are listed below

1. Run denoising of task data in 2 steps:
    * Denoise conversation data `~/code/clean.py`: This code was written by Zaid Zada (https://github.com/zaidzada/fconv/blob/main/code/clean.py). The confound regression model used for denoising in this data is `model9_task` which includes confound specified from model #9 in our confound modeling, as well as 7 additional task regressors: "trial_boxcar, prompt_boxcar, speech_boxcar, listen_boxcar, button_press, receive_press, screen_change". \

    * Denoise listening data `~/code/clean_black.py`: This uses the confound model #9 ONLY since this task is not conversation based! Run `~/code/slurm_clean_black.sh` to parallelize process. \

2. Dyadic hyperalignment and ISC (atlas based): '~/code/slurm_hyp_isc.sh`: 
    * Hyperalignment is a multivariate technique that learns how to map each participant’s neural response patterns into a common representational space. In a dyadic context, it identifies the shared information (e.g., the shared representational geometry) between two participants, transforming their brain data so they become functionally aligned. Once hyperaligned, each participant’s data is more directly comparable, facilitating subsequent intersubject correlation and other group-level analyses.\

    * Intersubject correlation (ISC) measures how similarly two (or more) participants’ brain signals fluctuate over time during the same task or stimulus. By correlating the neural timecourses of participant A with participant B, we can assess whether their brain activity patterns show consistent, synchronized responses. This helps identify shared neural dynamics that might be related to the stimulus or interaction at hand.\

    * The code in its current iteration was last edited (2/4/2026), and now will run hyperalignment+ISC for a specific model (model9 or model9_task) AND for a specific parcellation (Schaefer 100parcel 17network, harvard-oxford subcortical, and Glasser atlas). I added this edit because of reviewer comments wanting us to run our analyses such that it includes subcortical regions, has higher parcellation atlas that include language specific network, and to run our analysis with regular confound modeling (model9) and the confound + task regressors (model9_task). I recognize that currently the script for the different atlases `~/code/run_fastsrm_<atlas_name>.py` is not the most efficient way of doing this, but currently that's the way it is formatted. I'll eventually clean this to account for the inificiency. 

## Subject exclusions for main ISC study
'sub-001','sub-101':
'sub-011','sub-111': sub-011 does not have task-Black data so could not perform dyadic hyperalignment
'sub-012','sub-112': sub-011 does not have task-Black data so could not perform dyadic hyperalignment
'sub-019','sub-119':
'sub-068','sub-168':
'sub-071','sub-171': 

## Analysis steps run on conversation data
3. Compute time lagged intersubject correlation analyses (`~/code/analyze_results.ipynb`). The purpose of this this notebook is to do the following things:
    - combine the time locked ISC results from step 2 and save as a .csv the different results based on the combination of model/condition/parcellation run
    - run lagged-ISC: 
4. Univariate analyses`~code/univariate_analysis.ipynb`


## Notes of changes made to confound modeling code
I’ve adapted Zaid’s confound modeling code to run using the .nii functional data. You can find the test code in scratch here:
`~/work/CONV_ISC/conversation_pipeline/code`.\

I have it running so that it runs the confound modeling with the task confounds + model 9. It outputs the cleaned trials for each condition as well. I’ve added all this information to the read.\ 

I made a few key changes which I am listing below.\ 

•	code/constants.py\
o	expanded the list of subjects with interrupted scans to include strangers + friend subjects\
•	code/clean.py\
o	imported variables from `constants.py` includes `SUBS` since we want to run this on the full sample not just strangers.\
o	Added a new flag to the clean.py script, so that on command line you can specify the condition (generate, read, or something else)  you want to clean. So the new command to clean should be something like this: ` do python code/clean.py -m $model -c read; done`\
o	The `run_level_regression` & `trial_level_regression` functions now use the condition variable and will denoise that condition and saved file will include the condition information.\
o	File naming nomenclature change from `fsaverage6.h5` (since Zaid wrote this for .gii files) to `MNI152NLin2009cAsym_desc-{condition}.h5`. I’m keeping bids formatting here, but if you think it would be more helpful to name it differently let me know.\

•	code/util/subject.py
o	I modified the `get_timing`,`get_trials`, ` get_timinglog_boxcars` functions to handle the condition information.\
o	I modified the ` get_raw_bold` function to work with .nii files instead of .gii. The most important change is just how the date is being imported, reshaped, and stacked to be compatible with the clean.py script. \

## Tracking of changes made to pipeline
 - Note that you need to use conda activate fconv 

