import os
import argparse
import h5py
import joblib
from glob import glob
from os.path import join as opj

import numpy as np
from scipy.stats import pearsonr

from nilearn import image as nimg
from nltools.data import Brain_Data
from nltools.mask import expand_mask, roi_to_brain
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_harvard_oxford
from brainiak.funcalign.fastsrm import FastSRM
from brainiak.isc import isc
    

def grab_func_black(dyad_id, clean_dir):
    """
    Load the 'listening task' data (train data) for two participants in a dyad.
    Checks if runs have different lengths (TRs); if so, trims the longer run.
    
    Returns: np.ndarray of shape (n_voxels, n_TRs, n_subjects)
    """

    # Set paths to clean data
    sub1_path = glob(opj(clean_dir, f"sub-0{dyad_id}/model9/sub-0{dyad_id}_task-black_space-MNI*.h5"))[0]
    sub2_path = glob(opj(clean_dir, f"sub-1{dyad_id}/model9/sub-1{dyad_id}_task-black_space-MNI*.h5"))[0]

    bd_sub1 = Brain_Data(sub1_path)  # shape => (TR, Voxel)
    bd_sub2 = Brain_Data(sub2_path)  # shape => (TR, Voxel)

    # Transpose for fast SRM format
    data_sub1 = bd_sub1.data.T       # => now (Voxel, TR)
    data_sub2 = bd_sub2.data.T       # => now (Voxel, TR)

    # Compare time lengths, trim the longer run
    T1 = data_sub1.shape[1]
    T2 = data_sub2.shape[1]

    if T1 != T2:
        if T1 < T2:
            print(f"[INFO] Dyad {dyad_id}: Subject0 has {T1} TRs < Subject1 who has {T2} TRs.")
            print(f"[INFO] Trimming Subject1 from {T2} to {T1} TRs.")
            data_sub2 = data_sub2[:, :T1]
        else:
            print(f"[INFO] Dyad {dyad_id}: Subject1 has {T2} TRs < Subject0 who has {T1} TRs.")
            print(f"[INFO] Trimming Subject0 from {T1} to {T2} TRs.")
            data_sub1 = data_sub1[:, :T2]


    # Stack dyad data
    dyad_data = []
    dyad_data.append(data_sub1)
    dyad_data.append(data_sub2)

    print(f"[TRAIN] Dyad: {dyad_id}")
    print(f"[TRAIN] Final data shape: {np.shape(dyad_data)} = (n_vox, n_TR, n_subs)")
    return dyad_data

def grab_func_conv(dyad_id, condition, clean_dir, model):
    """
    Function for grabbing the conversation task data
    default model = 'model9_task' # this can also be model9 which does have additional task regressors
    """
    
    # Set path to task-Conv
    sub1_path = glob(opj(clean_dir,f"sub-0{dyad_id}/{model}/sub-0{dyad_id}_task-conv*{condition}*nii.gz"))[0]
    sub2_path = glob(opj(clean_dir, f"sub-1{dyad_id}/{model}/sub-1{dyad_id}_task-conv*{condition}*nii.gz"))[0]
    files = [sub1_path, sub2_path]

    sub_bd_list = []
    sub_data_list = []

    # Load each subject's data as a Brain_Data object and convert to (vox, time)
    for file in files:
        sub_img = nimg.load_img(file)
        sub_bd = Brain_Data(sub_img)        # shape => (time, vox) in sub_bd.data
        sub_bd_list.append(sub_bd)
        
        # Transpose to make shape => (vox, time)
        sub_data = sub_bd.data.T           # so we have (n_voxels, n_TRs)
        sub_data_list.append(sub_data)

    # Stack the 2D arrays along the last dimension => (n_voxels, n_TR, n_subjects)
    dyad_data = []
    dyad_data.append(sub_data_list[0])
    dyad_data.append(sub_data_list[1])

    vox_num, nTR, num_subs = np.shape(dyad_data)
    print(f"[TEST] Dyad {dyad_id}, condition {condition}")
    print(f"  Participants: {num_subs}")
    print(f"  Voxels per participant: {vox_num}")
    print(f"  TRs per participant: {nTR}")

    # Return both the Brain_Data objects and the stacked NumPy array
    return sub_bd_list, dyad_data


def main():
    """
    Main function to run FastSRM hyperalignment, reconstruction, and optionally ISC.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dyad_id', type=str, required=True,
                        help="The dyad ID (e.g. '04')")
    parser.add_argument('--conditions', type=str, nargs='+', default=['generate', 'read'],
                        help="Test conversation conditions (list).")
    parser.add_argument('--clean_dir', type=str, default='./data/derivatives/clean',
                        help="Directory containing cleaned data.")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help="Base directory containing data/atlases, etc.")
    parser.add_argument('--model', type=str, default='model9_task',
                        help="denoising model used for conversation task")
    parser.add_argument('--atlas_path', type=str, 
                        default='./data/atlases/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz.npy',
                        help="Path to the parcellation atlas (NIfTI).")
    parser.add_argument('--atlas_label', type=str, 
                        default='hosubcortical21p',
                        help="Label you want to use in output files.")
    parser.add_argument('--n_components', type=int, default=20,
                        help="Number of shared components for SRM.")
    parser.add_argument('--n_iter', type=int, default=10,
                        help="Number of iterations for SRM.")
    parser.add_argument('--save_dir', type=str, default='hyperaligned',
                        help="Directory to store results (e.g., SRM temporary files, recon data).")
    parser.add_argument('--do_isc', action='store_true',
                        help="If set, also compute ISC after reconstruction.")
    args = parser.parse_args()
    
    # args.clean_dir = os.path.expanduser(args.clean_dir)
    # args.data_dir = os.path.expanduser(args.data_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # 1. Load your training data for hyperalignment
    # --------------------------------------------------------------------------
    print(f"[INFO] Loading hyperalignment task data for dyad:{args.dyad_id} using {args.atlas_label} parcellation atlas")
    train_data = grab_func_black(args.dyad_id, args.clean_dir)
    n_train = len(train_data) # number of training subjects

    # --------------------------------------------------------------------------
    # 2. Initialize and fit FastSRM on the training data
    # --------------------------------------------------------------------------
    print("[INFO] Initializing SRM with n_components:", args.n_components)
    
    fastsrm = FastSRM(
        atlas=args.atlas_path,
        n_components=args.n_components,
        n_iter=args.n_iter,
        n_jobs=1,
        low_ram=True,
        aggregate="mean",  # Use mean for training
        temp_dir="./code/hyperaligned"
    )
    print("[INFO] Fitting SRM on training data...")
    fastsrm.fit(train_data)  # train_data shape: (n_voxels, n_TRs, n_subs)
    
    model_outfile = opj(args.save_dir, f"hyperaligned/srm_task-Black/fastsrm_dyad_{args.dyad_id}_{args.model}_{args.atlas_label}.joblib")
    joblib.dump(fastsrm, model_outfile)
    print(f"[INFO] Saved fitted FastSRM to {model_outfile}")
    # --------------------------------------------------------------------------
    # 3. Transform the TRAIN data into the shared space (optional, for checking)
    # --------------------------------------------------------------------------
    print("[INFO] Projecting TRAIN data into shared space...")
    train_shared_response = fastsrm.transform(train_data) #(n_components, n_TRs)
    
    # Initialize a counter to track the cumulative number of subjects in the model.
    cumulative_subjects = n_train

    # --------------------------------------------------------------------------
    # 4. Process each conversation condition (test data) separately.
    # Since the same two subjects are used, we always use subjects_indexes = [0, 1]
    # --------------------------------------------------------------------------
    
    # For test data processing, update the model so that transform returns subject-specific responses.
    fastsrm.aggregate = None

    condition_results = {}
    for cond in args.conditions:
        print(f"\n[INFO] Processing test condition: {cond}")
        sub_bd_list, test_data = grab_func_conv(args.dyad_id, cond, args.clean_dir,args.model)
        # test_data is a list of arrays, one per subject, each of shape [n_voxels, n_TRs_test] (e.g., 1200 TRs)

        # Transform the test data into the shared space.
        # With aggregate=None, fastsrm.transform returns a list of subject-specific shared responses.
        test_shared_list = fastsrm.transform(test_data)

        print("[INFO] Reconstructing test data back to voxel space...")
        # Since the test subjects are the same as the training subjects, we reconstruct using subjects_indexes = [0, 1]
        test_reconstructed_list = fastsrm.inverse_transform(
            test_shared_list,
            subjects_indexes=[0, 1]
        )
            
        # Save or store the reconstructed data for this condition.
        condition_results[cond] = test_reconstructed_list
        # ----------------------------------------------------------------------
        # 5. Save the reconstructed hyperaligned data
        # ----------------------------------------------------------------------
        out_path = opj(args.save_dir, f"hyperaligned/srm_task-Conv/dyad_{args.dyad_id}_cond_{cond}_{args.model}_{args.atlas_label}.h5")
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("bold", data=test_reconstructed_list)
        print(f"[INFO] Hyperaligned conversation data saved to: {out_path}")

        # ----------------------------------------------------------------------
        # 5. (Optional) Compute ISC for the reconstructed test data
        # ----------------------------------------------------------------------
        if args.do_isc:
            print("[INFO] Computing ISC using parcellation atlas (ROI-based)...")

            # -- Step 6a: Load or fetch an atlas and expand to ROIs --
            #atlas = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17)  # Adjust n_rois if needed
            atlas = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")  # Adjust n_rois if needed

            atlas_img = Brain_Data(atlas['maps'])  # NIfTI atlas
            roi_masks = expand_mask(atlas_img)  # list of Brain_Data masks
            
            sub1_rec_conv = sub_bd_list[0].copy() 
            sub2_rec_conv = sub_bd_list[1].copy()
            sub1_rec_conv.data = test_reconstructed_list[0].T
            sub2_rec_conv.data = test_reconstructed_list[1].T
            
            # Apply mask
            sub1_rec_conv_masked = sub1_rec_conv.extract_roi(atlas_img)
            sub2_rec_conv_masked = sub2_rec_conv.extract_roi(atlas_img)
            
            # Save masked data
            sub1_outpath = opj(args.save_dir, f"hyperaligned/srm_task-Conv/sub-0{args.dyad_id}_cond_{cond}_{args.model}_{args.atlas_label}.h5")
            sub2_outpath = opj(args.save_dir, f"hyperaligned/srm_task-Conv/sub-1{args.dyad_id}_cond_{cond}_{args.model}_{args.atlas_label}.h5")
            with h5py.File(sub1_outpath, 'w') as hf:
                hf.create_dataset("bold", data=sub1_rec_conv_masked)
            with h5py.File(sub2_outpath, 'w') as hf:
                hf.create_dataset("bold", data=sub2_rec_conv_masked)
            
            n_TR_total = sub1_rec_conv_masked.shape[1]  # 1200 since each run is 120 TRs and we have 10
            n_parcels = sub1_rec_conv_masked.shape[0]   # e.g. 100 if using a 100-ROI atlas

            n_trials = 10
            trial_len = 120
            print(f"Number of: TRs = {n_TR_total}, parcels =  {n_parcels}, trials = {n_trials}, trial length = {trial_len}") # Sanity check 

            # trim onset and end
            trim_start = 7
            trim_end   = 7
            trimmed_len = trial_len - (trim_start + trim_end)  # e.g. 120 - 14 = 106
            
            # store ISC for each ROI x trial
            isc_array = np.zeros((n_parcels, n_trials))  
            
            for i in range(n_trials):
                # Calculate block indices
                block_start = i * trial_len
                block_end   = block_start + trial_len

                # Trim the start/end from that block
                sub1_block = sub1_rec_conv_masked[:, block_start:block_end]  # shape (120, n_parcels)
                sub2_block = sub2_rec_conv_masked[:, block_start:block_end]

                # Slice off the first and last 7 TRs 
                sub1_block_trimmed = sub1_block[:, trim_start : trial_len - trim_end]  # shape => (106, n_parcels)
                sub2_block_trimmed = sub2_block[:, trim_start : trial_len - trim_end]

                # Prepare data for BrainIAK's `isc()` => (n_TRs, n_voxels, n_subs)
                roi_stacked = np.stack([sub1_block_trimmed, sub2_block_trimmed], axis=0)
                roi_stacked = np.transpose(roi_stacked, (2, 1, 0))
                # LMT to do------ add in sanity check here for the correct dimensions and specify spatial vs temporal ISC. Currently set for temporal!!
                
                # Compute dyadic ISC
                isc_map = isc(roi_stacked, pairwise=False)
                isc_array[:, i] = isc_map

                # Print some summary stats. LMT-- it would be nice to have this output as a result file but for now keeping here. 
                mean_isc_per_trial = isc_array.mean(axis=0)  # average across parcels
                print(f"[INFO] Mean ISC per trial: {mean_isc_per_trial}")
                overall_mean_isc = isc_array.mean()
                print(f"[INFO] Overall mean ISC across all parcels & trials: {overall_mean_isc}")

                # Save or print the ROI-based ISC results
                isc_out_path = opj(args.save_dir, f"isc/dyad_{args.dyad_id}_cond_{cond}_ISC_{args.model}_{args.atlas_label}.h5")
                with h5py.File(isc_out_path, 'w') as hf:
                    hf.create_dataset("isc", data=isc_array)
                    hf.attrs["n_parcels"] = n_parcels
                    hf.attrs["n_trials"] = n_trials
                    hf.attrs["trim_start"] = trim_start
                    hf.attrs["trim_end"] = trim_end

                print(f"[INFO] ISC results for trial {i} saved to {isc_out_path}")


if __name__ == "__main__":
    main()

