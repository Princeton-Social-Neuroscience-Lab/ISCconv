import subprocess
import pandas as pd
import numpy as np
from glob import glob
import shutil
import os
from os.path import join as opj
import nilearn
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import concat_imgs, mean_img, load_img, index_img, math_img 
from nilearn.plotting import plot_design_matrix, plot_glass_brain, plot_contrast_matrix, plot_stat_map
from nilearn.glm import threshold_stats_img
from scipy.stats import norm
import matplotlib.pyplot as plt
from nilearn.glm.second_level import SecondLevelModel
from nilearn.reporting import get_clusters_table
# from atlasreader import create_output
#from utilities import ensure_dir
import json
import time
from os.path import basename, exists, join, splitext
# from extract_counfounds import extract_confounds
import joblib
from joblib import Parallel, delayed

def ensure_dir(ed):
    import os
    try:
        os.makedirs(ed)
    except OSError:
        if not os.path.isdir(ed):
            raise

def pad_vector(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

def ensure_dir(folder):
    if os.path.isdir(folder) == False:
        os.makedirs(folder)
        
def make_dm_sam(mc, tr):
    z_mc = zscore(mc)
    z_mc.fillna(value=0, inplace=True)
    return Design_Matrix(z_mc, sampling_freq=1/tr)

# Function for loading in confounds files
def load_confounds(confounds_fn):

    # Load the confounds TSV files
    confounds_df = pd.read_csv(confounds_fn, sep='\t')

    # Load the JSON sidecar metadata
    with open(splitext(confounds_fn)[0] + '.json') as f:
        confounds_meta = json.load(f)

    return confounds_df, confounds_meta
    
#----------------------- SET UP
# Get subject id information for each conversation partner
hyp_dir = './data/hyperalignment'
subjlist_all = [i for i in os.listdir(hyp_dir) if i.startswith('sub-')]
subjlist_all.sort()

# Exclude data from the dyads that don't have movie data
exclude_list = ['sub-001','sub-101','sub-011','sub-111','sub-012','sub-112',
                'sub-019','sub-119','sub-068','sub-168','sub-071','sub-171']
subs = [e for e in subjlist_all if e not in exclude_list]


#directories
base_dir='./univariate_analysis/'
prep_dir = './data/bids/derivatives/fmriprep/'
home_dir = './data'
deno_dir = './data/denoised'
hyp_dir = './data/hyperalignment'
output_dir='./univariate_analysis'
raw_dir = './data/bids/'
write_dir=opj(base_dir,'Firstlevel')
ensure_dir(write_dir)

tr = 1.5
runs = [1,2,3,4,5]

#--------------------------------- univariate model specification

def univariate_modeling(sub):
    design_matrices=[]
    fmri_img=[]
    for run in runs:
        print('run: %s' % run)
        start = time.time()
    
        # load funcally aligned file in mni space
        func_mni = glob(opj(prep_dir,sub,'ses-1','func', f'*Conv*run-{run}*MNI*preproc_bold*gz'))[0]
        
        # get 
       
        # func_mni = glob(opj(deno_dir,sub, f'*Conv*run-{run}*aligned*nii'))[0]

        if not os.path.isfile(func_mni):
            print('missing func')
            continue
        func1_img =load_img(func_mni)
        n_frames=func1_img.shape[3]
        frame_times = np.arange(n_frames) * tr  

        
        confounds = pd.read_csv(glob(os.path.join(home_dir,'denoised',sub,'func',f'*Conv*run-{run}*model9_timeseries.csv'))[0]).fillna(value=0, inplace=True)

        # onest file 
        events = pd.read_csv(glob(opj(home_dir,'CONV_csv','CONV_csv','onsets_correct', f"{sub}*run_{run}.csv"))[0])
        events.drop(['Unnamed: 0'],axis=1, inplace=True)

    
         # create design matrix 
        dm_conv = make_first_level_design_matrix(
            frame_times,
            events,
            add_regs=confounds,
            # add_reg_names=list(confounds.columns),
            hrf_model='spm',
            drift_model='polynomial',
            drift_order=4,     
            )
        design_matrices.append(dm_conv)
        fmri_img.append(func1_img)
    
        # check if design matrices are the same shape
        dm_shapes = []
        dm_cols = []
        for dmnr in range(len(design_matrices)):
            dm_shapes.append(design_matrices[dmnr].shape)
            dm_cols.append(design_matrices[dmnr].shape[1])
        
        if len(np.unique(dm_cols)) > 1:
            print('unequal design matrices')
            concat_dms = pd.concat(design_matrices).fillna(0)
            # now split them up again
            design_matrices_new = []
            for dmnr in range(len(design_matrices)):
                concat_dms.iloc
                design_matrices_new.append(concat_dms.iloc[((dmnr) *dm_shapes[0][0]):((dmnr + 1) *dm_shapes[0][0])])
            design_matrices = design_matrices_new.copy()   
            
            
        print('Length of img check' + str(len(fmri_img)))
        print('Length of design_matrices check' + str(len(design_matrices)))
    
        fmri_glm = FirstLevelModel(minimize_memory=True, standardize=False, smoothing_fwhm=5)#, signal_scaling=False)
        fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices) # might have to turn off signal scaling because data might be centered
    
        end = time.time()

        n_columns = design_matrices[0].shape[1]

        contrasts = {'R_vs_G': pad_vector([-1, 1], n_columns),
                     'G_vs_R': pad_vector([1, -1], n_columns),
                     'R': pad_vector([0, 1], n_columns),
                     'G': pad_vector([1, 0], n_columns)}


        fig, axes = plt.subplots(nrows=1, ncols=4)

        for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
            print('  Contrast % 2i out of %i: %s' % (
                index + 1, len(contrasts), contrast_id))
                # estimate the contasts
                # note that the model implictly compute a fixed effects across the two sessions
            z_map = fmri_glm.compute_contrast(
                contrast_val, output_type='z_score')

             #fdr thresholding
            clean_map, threshold = threshold_stats_img(
            z_map, alpha=.05, height_control='fdr', cluster_threshold=0)

            # Write the resulting stat images to file
            ensure_dir(opj(write_dir,contrast_id))
            z_image_path = opj(write_dir,contrast_id, '%s_%s_model-9_z_map.nii.gz' % (sub, contrast_id))
            z_map.to_filename(z_image_path)

            # plot_glass_brain(z_map, colorbar=False, threshold=norm.isf(0.001),
            #                              title='%s, subject %s, uncorr at p<0.001' % (contrast_id, sub),
            #                               axes=axes[index],#, int(midx % 4)],
            #                               plot_abs=False, display_mode='x')
            # fig.suptitle('Contrasts subject %s, uncorr at p<0.001' % (sub))
            # plt.show()
            print("Time consumed in working: ",end - start)

#----------------------------- RUN
#testing

start = time.time()
# n_jobs is the number of parallel jobs
Parallel(n_jobs=8, backend='loky')(delayed(univariate_modeling)(sub) for sub in subs)
end = time.time()
print('{:.4f} s'.format(end-start))
