"""
Filename : feature_extraction_ABIDE.py
Created Date : 20/05/2020
Last Edited : 29/06/2020
Author : Mateo MAHAUT (mmahaut@ensc.fr) 
Git : https://github.com/mahautm/INT_fMRI_processing.git

"""

import os
import sys
import json
import shutil
import glob
import nibabel as nib
import nibabel.gifti as ng
import scipy.io as scio
import numpy as np
import time
from scipy.stats import pearsonr


def extract_all_abide(
    subject_list="./url_preparation/subs_list.json",
    data_list_files="./url_preparation/files_to_download.json",
    raw_data_path="./raw_data_ABIDE",
    force_destination_folder=False,
    template="fsaverage5",
    processed_data_path="./processed_ABIDE",
    matlab_runtime_path="/usr/local/MATLAB/MATLAB_Runtime/v95",
    matlab_script_path="./for_redistribution_files_only",
):
    """
    calls all feature-extraction functions, in order, on all subjects.
    requires freesurfer to be setup (tested on v6.0.0-a), as well as FSL and Matlab runtime (v95)

    Parameters
    ----------

    subject_list : string, optional ("./url_preparation/subs_list.json" by default)
        is a path to a JSON file listing all subjects to call function on 

    data_list_files : string, optional ("./url_preparation/files_to_download.json" by default)
        is a path to a JSON file, where required freesurfer data, and fMRI modalities are found
    
    raw_data_path : string, optional ("./raw_data_ABIDE" by default)
        is the path to the folder where all aquired data will be saved
    
    force_destination_folder :  boolean, optional, default False
        is a bool that if activated will force SUBJECTS_DIR to be destination folder during registration
    
    template : ("fsaverage5" by default)
        must be found in freesurfer's SUBJECTS_DIR, used during freesurfer operations
    
    processed_data_path : string, optional ("./processed_ABIDE" by default)
        where final processed data should be placed

    matlab_runtime_path : string, optional ("/usr/local/MATLAB/MATLAB_Runtime/v95" by default)
        path to the folder where Runtime's bin folder is found, used for the gyrification computing
    
    matlab_script_path : string, optional ("./for_redistribution_files_only" by default)
        path to the folder where the .sh run_find_eig.sh and find_eig are found, used for gyrification computing


    Notes : 
    -----
    complet ABIDE subject list may be found here :
    https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/yamls/subs_list.yml
    """
    # opening .json
    subs_list_file = open(subject_list)
    subs_list = json.load(subs_list_file)
    data_list_file = open(data_list_files)
    data_list = json.load(data_list_file)

    for subject in subs_list:
        extract_one_abide(
            subject,
            data_list,
            raw_data_path,
            force_destination_folder,
            template,
            processed_data_path,
            matlab_runtime_path,
            matlab_script_path,
        )

    subs_list_file.close()
    data_list_file.close()


def extract_one_abide(
    subject,
    ref_subject,
    data_list,
    raw_data_path="/scratch/mmahaut/data/abide/downloaded_preprocessed",
    force_destination_folder=False,
    template="fsaverage5",
    contrast="t1",
    rfmri_features_data_path="/scratch/mmahaut/data/abide/features_rsfMRI",
    gyrification_features_data_path="/scratch/mmahaut/data/abide/features_gyrification",
    matlab_runtime_path="/scratch/mmahaut/tools/MATLAB_Runtime/v95",
    matlab_script_path="/scratch/mmahaut/scripts/INT_fMRI_processing/for_redistribution_files_only",
    intermediary_data_path="/scratch/mmahaut/data/abide/intermediary",
):
    """
    Grouping of all functions required to produce the rsfMRI features as a correlation matrix between voxels and ROIs
    and the gyrification features as a eigen vector matrix. All missing required subject data will be downloaded.
    will save the correlation matrix directly to designated folder, and all intermediary steps

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    data_list : a dictionary folowing this architecture
        ['rsfMRI']

            ['pipeline'] = {"ccs","cpac","dparsf","niak"} 
            ['strategy'] = {"filt_global","filt_noglobal","nofilt_global","nofilt_noglobal"}
            ['file identifier'] = the FILE_ID value from the summary spreadsheet
            ['derivative'] = {"alff","degree_binarize","degree_weighted","dual_regression",
               "eigenvector_binarize","eigenvector_weighted","falff","func_mask",
               "func_mean","func_preproc","lfcd","reho","rois_aal","rois_cc200",
               "rois_cc400","rois_dosenbach160","rois_ez","rois_ho","rois_tt","vmhc"}

        ['freesurfer']
            ['labels'] = table of files to download
            ['mri'] = table of files to download
            ['scripts'] = table of files to download
            ['surf'] = table of files to download
            ['stats'] = table of files to download
    
    raw_data_path : string path to file ("/scratch/mmahaut/data/abide/downloaded_preprocessed" by default)
        is the path to the folder where all original preprocessed data will be downloaded
    
    force_destination_folder :  boolean, optional, default False
        is a bool that if activated will force SUBJECTS_DIR to be destination folder during registration
    
    template : string, optional ("fsaverage5" by default)
        name of the template used bu freesurfer functions, must be found in freesurfer's SUBJECTS_DIR

    contrast : {"t1", "dti", "t2", "bold"}
        used by freesurfer's bbregister
    
    rfmri_features_data_path : string, optional ("/scratch/mmahaut/data/abide/features_rsfMRI" is default)
        path to the file where the correlation matrix will be saved. If the path does not exist, folders will be added.

    gyrification_features_data_path : string, optional ("/scratch/mmahaut/data/abide/features_gyrification" is default)
        path to the file where the gyrification eigen-vector matrix will be saved. If the path does not exist, folders will be added.

    matlab_runtime_path : string path to file ("/usr/local/MATLAB/MATLAB_Runtime/v95" by default)
         path to the folder where Runtime's bin folder is found, used for the gyrification computing
   
    matlab_script_path : string path to file ("./for_redistribution_files_only" by default)
        path to the folder where the .sh run_find_eig.sh and find_eig are found, used for gyrification computing

    intermediary_data_path : string, optional ("/scratch/mmahaut/data/abide/intermediary" is default)
        path to the file where all intermediary data will be saved, included registered rsfMRI and temporally splitted surface data


    Notes : 
    -----
    requires freesurfer to be setup (tested on v6.0.0-a), as well as FSL and matlab runtime (v95)

    complet ABIDE subject list may be found here :
    https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/yamls/subs_list.yml
    """
    download_abide_urls(subject, data_list, raw_data_path)
    compute_rfMRI_features(
        subject,
        data_list,
        raw_data_path,
        rfmri_features_data_path,
        intermediary_data_path,
        contrast,
        force_destination_folder,
        template,
    )
    compute_gyrification_features(
        subject,
        ref_subject,
        raw_data_path,
        gyrification_features_data_path,
        intermediary_data_path,
        matlab_runtime_path,
        matlab_script_path,
        template,
    )


def extract_one_interTVA(
    subject,
    ref_subject,
    data_list,
    raw_data_path,
    template="fsaverage5",
    gyrification_features_data_path="/scratch/mmahaut/data/intertva/features_gyrification",
    matlab_runtime_path="/scratch/mmahaut/tools/MATLAB_Runtime/v95",
    matlab_script_path="/scratch/mmahaut/scripts/INT_fMRI_processing/for_redistribution_files_only",
    intermediary_data_path="/scratch/mmahaut/data/intertva/intermediary",
):
    """
    Grouping of all functions required to produce the rsfMRI features as a correlation matrix between voxels and ROIs
    and the gyrification features as a eigen vector matrix. All missing required subject data will be downloaded.
    will save the correlation matrix directly to designated folder, and all intermediary steps

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    data_list : a dictionary folowing this architecture
        ['rsfMRI']

            ['pipeline'] = {"ccs","cpac","dparsf","niak"} 
            ['strategy'] = {"filt_global","filt_noglobal","nofilt_global","nofilt_noglobal"}
            ['file identifier'] = the FILE_ID value from the summary spreadsheet
            ['derivative'] = {"alff","degree_binarize","degree_weighted","dual_regression",
               "eigenvector_binarize","eigenvector_weighted","falff","func_mask",
               "func_mean","func_preproc","lfcd","reho","rois_aal","rois_cc200",
               "rois_cc400","rois_dosenbach160","rois_ez","rois_ho","rois_tt","vmhc"}

        ['freesurfer']
            ['labels'] = table of files to download
            ['mri'] = table of files to download
            ['scripts'] = table of files to download
            ['surf'] = table of files to download
            ['stats'] = table of files to download
    
    raw_data_path : string path to file ("/scratch/mmahaut/data/abide/downloaded_preprocessed" by default)
        is the path to the folder where all original preprocessed data will be downloaded
    
    force_destination_folder :  boolean, optional, default False
        is a bool that if activated will force SUBJECTS_DIR to be destination folder during registration
    
    template : string, optional ("fsaverage5" by default)
        name of the template used bu freesurfer functions, must be found in freesurfer's SUBJECTS_DIR
    
    rfmri_features_data_path : string, optional ("/scratch/mmahaut/data/abide/features_rsfMRI" is default)
        path to the file where the correlation matrix will be saved. If the path does not exist, folders will be added.

    gyrification_features_data_path : string, optional ("/scratch/mmahaut/data/abide/features_gyrification" is default)
        path to the file where the gyrification eigen-vector matrix will be saved. If the path does not exist, folders will be added.

    matlab_runtime_path : string path to file ("/usr/local/MATLAB/MATLAB_Runtime/v95" by default)
         path to the folder where Runtime's bin folder is found, used for the gyrification computing
   
    matlab_script_path : string path to file ("./for_redistribution_files_only" by default)
        path to the folder where the .sh run_find_eig.sh and find_eig are found, used for gyrification computing

    intermediary_data_path : string, optional ("/scratch/mmahaut/data/abide/intermediary" is default)
        path to the file where all intermediary data will be saved, included registered rsfMRI and temporally splitted surface data


    Notes : 
    -----
    requires freesurfer to be setup (tested on v6.0.0-a), as well as FSL and matlab runtime (v95)

    complet ABIDE subject list may be found here :
    https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/yamls/subs_list.yml
    """
    download_interTVA_from_frioul(subject, data_list, raw_data_path)
    compute_gyrification_features(
        subject,
        ref_subject,
        raw_data_path,
        gyrification_features_data_path,
        intermediary_data_path,
        matlab_runtime_path,
        matlab_script_path,
        template,
    )


def compute_rfMRI_features(
    subject,
    data_list,
    raw_data_path,
    processed_data_path,
    intermediary_data_path,
    contrast="t1",
    force_destination_folder=False,
    template="fsaverage5",
):
    """
    Grouping of all functions required to produce the rsfMRI features as a correlation matrix between voxels and ROIs
    will save the correlation matrix directly to designated folder, and all intermediary steps

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    data_list : a dictionary folowing this architecture
        ['rsfMRI']

            ['pipeline'] = {"ccs","cpac","dparsf","niak"} 
            ['strategy'] = {"filt_global","filt_noglobal","nofilt_global","nofilt_noglobal"}
            ['file identifier'] = the FILE_ID value from the summary spreadsheet
            ['derivative'] = {"alff","degree_binarize","degree_weighted","dual_regression",
               "eigenvector_binarize","eigenvector_weighted","falff","func_mask",
               "func_mean","func_preproc","lfcd","reho","rois_aal","rois_cc200",
               "rois_cc400","rois_dosenbach160","rois_ez","rois_ho","rois_tt","vmhc"}

        ['freesurfer']
            ['labels'] = table of files to download
            ['mri'] = table of files to download
            ['scripts'] = table of files to download
            ['surf'] = table of files to download
            ['stats'] = table of files to download
    
    contrast : {"t1", "dti", "t2", "bold"}
        used by freesurfer's bbregister

    raw_data_path : string path to file ("/scratch/mmahaut/data/abide/downloaded_preprocessed" by default)
        is the path to the folder where subject data will be found

    force_destination_folder :  boolean, optional, default False
        is a bool that if activated will force SUBJECTS_DIR to be destination folder during registration
    
    template : string, optional ("fsaverage5" by default)
        name of the template used bu freesurfer functions, must be found in freesurfer's SUBJECTS_DIR

    processed_data_path : string, optional ("/scratch/mmahaut/data/abide/features_rsfMRI" is default)
        path to the file where the correlation matrix will be saved. If the path does not exist, folders will be added.

    intermediary_data_path : string, optional ("/scratch/mmahaut/data/abide/intermediary" is default)
        path to the file where all intermediary data will be saved, included registered rsfMRI and temporally splitted surface data
    

    """
    minimally_preproc_register(
        subject,
        data_list["rsfMRI"]["derivative"],
        raw_data_path,
        intermediary_data_path,
        force_destination_folder,
        contrast,
    )
    split_dim_time(
        subject, data_list["rsfMRI"]["derivative"], intermediary_data_path,
    )
    check_and_project_vol2surf(
        subject,
        data_list["rsfMRI"]["derivative"],
        raw_data_path,
        out_dir=intermediary_data_path,
        template=template,
    )
    check_and_correlate(
        subject, template, raw_data_path, intermediary_data_path, processed_data_path
    )


def compute_rfMRI_features_interTVA(
    subject,
    data_list,
    raw_data_path="/scratch/mmahaut/data/abide/downloaded_preprocessed",
    template="fsaverage5",
    processed_data_path="/scratch/mmahaut/data/abide/features_rsfMRI",
    intermediary_data_path="/scratch/mmahaut/data/abide/intermediary",
):
    """
    Grouping of all functions required to produce the rsfMRI features as a correlation matrix between voxels and ROIs
    will save the correlation matrix directly to designated folder, and all intermediary steps

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    data_list : a dictionary folowing this architecture
        ['rsfMRI']

            ['pipeline'] = {"ccs","cpac","dparsf","niak"} 
            ['strategy'] = {"filt_global","filt_noglobal","nofilt_global","nofilt_noglobal"}
            ['file identifier'] = the FILE_ID value from the summary spreadsheet
            ['derivative'] = {"alff","degree_binarize","degree_weighted","dual_regression",
               "eigenvector_binarize","eigenvector_weighted","falff","func_mask",
               "func_mean","func_preproc","lfcd","reho","rois_aal","rois_cc200",
               "rois_cc400","rois_dosenbach160","rois_ez","rois_ho","rois_tt","vmhc"}

        ['freesurfer']
            ['labels'] = table of files to download
            ['mri'] = table of files to download
            ['scripts'] = table of files to download
            ['surf'] = table of files to download
            ['stats'] = table of files to download

    raw_data_path : string path to file ("/scratch/mmahaut/data/abide/downloaded_preprocessed" by default)
        is the path to the folder where subject data will be found

    force_destination_folder :  boolean, optional, default False
        is a bool that if activated will force SUBJECTS_DIR to be destination folder during registration
    
    template : string, optional ("fsaverage5" by default)
        name of the template used bu freesurfer functions, must be found in freesurfer's SUBJECTS_DIR

    processed_data_path : string, optional ("/scratch/mmahaut/data/abide/features_rsfMRI" is default)
        path to the file where the correlation matrix will be saved. If the path does not exist, folders will be added.

    intermediary_data_path : string, optional ("/scratch/mmahaut/data/abide/intermediary" is default)
        path to the file where all intermediary data will be saved, included registered rsfMRI and temporally splitted surface data
    

    """

    check_and_project_vol2surf(
        subject,
        data_list["rsfMRI"]["derivative"],
        raw_data_path,
        out_dir=intermediary_data_path,
        template=template,
    )
    check_and_correlate(
        subject, template, raw_data_path, intermediary_data_path, processed_data_path
    )


def compute_gyrification_features(
    subject,
    ref_subject,
    raw_data_path,
    processed_data_path,
    intermediary_data_path="/scratch/mmahaut/data/abide/intermediary",
    matlab_runtime_path="/usr/local/MATLAB/MATLAB_Runtime/v95",
    matlab_script_path="./for_redistribution_files_only",
    template="fsaverage5",
):
    """
    Grouping all functions required to produce the gyrification features as a eigen vector matrix
    will save the correlation matrix directly to designated folder, and all intermediary steps

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    raw_data_path : string path to file ("/scratch/mmahaut/data/abide/downloaded_preprocessed" by default)
        is the path to the folder where subject data will be found

    processed_data_path : string, optional ("/scratch/mmahaut/data/abide/features_gyrification" is default)
        path to the file where the gyrification eigen-vector matrix will be saved. If the path does not exist, folders will be added.
    
    matlab_runtime_path : string path to file ("/usr/local/MATLAB/MATLAB_Runtime/v95" by default)
         path to the folder where Runtime's bin folder is found, used for the gyrification computing
   
    matlab_script_path : string path to file ("./for_redistribution_files_only" by default)
        path to the folder where the .sh run_find_eig.sh and find_eig are found, used for gyrification computing

    intermediary_data_path : string, optional ("/scratch/mmahaut/data/abide/intermediary" is default)
        path to the file where all intermediary data will be saved, included registered rsfMRI and temporally splitted surface data

    """
    prepare_matlab(subject, raw_data_path, intermediary_data_path)
    matlab_find_eig(
        subject,
        intermediary_data_path + "/" + subject,
        matlab_runtime_path,
        matlab_script_path,
    )
    align_gyrification(subject, intermediary_data_path, template),
    gyrification_sign(
        subject, ref_subject, processed_data_path, intermediary_data_path, template
    )


def download_interTVA_from_frioul(subject, data_list, destination_folder):
    """
    Parameters : 
    ----------
    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    data_list : a dictionary folowint this architecture
        ['rsfMRI']

            ['pipeline'] = {"ccs","cpac","dparsf","niak"} 
            ['strategy'] = {"filt_global","filt_noglobal","nofilt_global","nofilt_noglobal"}
            ['file identifier'] = the FILE_ID value from the summary spreadsheet
            ['derivative'] = {"alff","degree_binarize","degree_weighted","dual_regression",
               "eigenvector_binarize","eigenvector_weighted","falff","func_mask",
               "func_mean","func_preproc","lfcd","reho","rois_aal","rois_cc200",
               "rois_cc400","rois_dosenbach160","rois_ez","rois_ho","rois_tt","vmhc"}

        ['freesurfer']
            ['labels'] = table of files to download
            ['mri'] = table of files to download
            ['scripts'] = table of files to download
            ['surf'] = table of files to download
            ['stats'] = table of files to download

    destination_folder : string path to file ("/scratch/mmahaut/data/intertva/downloaded_preprocessed" by default)
        is the path to the folder where all original preprocessed data will be downloaded

    Notes :
    -----

    """
    # Adding freesurfer directory
    for key in data_list["freesurfer"]:
        path = "{}/{}/{}".format(destination_folder, subject, key)
        if not os.path.exists(path):
            os.makedirs(path)
        for file in data_list["freesurfer"][key]:
            if not os.path.exists(
                "{0}/{1}/{2}/{3}".format(destination_folder, subject, key, file)
            ):
                cmd = "scp mahaut.m@frioul.int.univ-amu.fr:/envau/work/banco/data/mri/InterTVA/my_intertva/surf/data/{1}/fs/{1}/{2}/{3} {0}/{1}/{2}/{3}".format(
                    destination_folder, subject, key, file
                )
                os.system(cmd)
            else:
                print(
                    "{} allready exists. Remove file if you wish to download again.".format(
                        file
                    )
                )
    print("Downloaded all of {}'s required files".format(subject))


def download_abide_urls(
    subject, data_list, destination_folder,
):
    """
    Here we build the urls, each file is then aquired and put in the right folder using wget 

    Parameters : 
    ----------
    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    data_list : a dictionary folowint this architecture
        ['rsfMRI']

            ['pipeline'] = {"ccs","cpac","dparsf","niak"} 
            ['strategy'] = {"filt_global","filt_noglobal","nofilt_global","nofilt_noglobal"}
            ['file identifier'] = the FILE_ID value from the summary spreadsheet
            ['derivative'] = {"alff","degree_binarize","degree_weighted","dual_regression",
               "eigenvector_binarize","eigenvector_weighted","falff","func_mask",
               "func_mean","func_preproc","lfcd","reho","rois_aal","rois_cc200",
               "rois_cc400","rois_dosenbach160","rois_ez","rois_ho","rois_tt","vmhc"}

        ['freesurfer']
            ['labels'] = table of files to download
            ['mri'] = table of files to download
            ['scripts'] = table of files to download
            ['surf'] = table of files to download
            ['stats'] = table of files to download

    destination_folder : string path to file ("/scratch/mmahaut/data/abide/downloaded_preprocessed" by default)
        is the path to the folder where all original preprocessed data will be downloaded

    Notes :
    -----
    urls are built as described by ABIDE documentation here : 
    http://preprocessed-connectomes-project.org/abide/download.html
    wget is called with '-c', so that no file is downloaded more than once, 
    this may allow to continue partial downloads too
    """

    # Adding rsFMRI file
    cmd = (
        "wget -c -q -P {}/{} https://s3.amazonaws.com/fcp-indi/data/Projects/"
        + "ABIDE_Initiative/Outputs/{}/{}/{}/{}_{}.nii.gz"
    ).format(
        destination_folder,
        subject,
        data_list["rsfMRI"]["pipeline"],
        data_list["rsfMRI"]["strategy"],
        data_list["rsfMRI"]["derivative"],
        subject,
        data_list["rsfMRI"]["derivative"],
    )
    os.system(cmd)

    # Adding minimally preprocessed file for registration
    cmd = (
        "wget -c -q -P {}/{} https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/func_minimal/{}_func_minimal.nii.gz"
    ).format(destination_folder, subject, subject,)
    os.system(cmd)

    # Adding freesurfer directory
    for key in data_list["freesurfer"]:
        path = "{}/{}/{}".format(destination_folder, subject, key)
        if not os.path.exists(path):
            os.makedirs(path)
        for file in data_list["freesurfer"][key]:
            cmd = (
                "wget -c -q -P {}/{}/{} https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/"
                + "{}/{}/{} "
            ).format(destination_folder, subject, key, subject, key, file)
            os.system(cmd)
    print("Downloaded all of {}'s required files".format(subject))


def register(
    subject, derivative, subject_folder, out_data, change_sub_dir=False, contrast="t1",
):
    """
    !! This function is NOT used in the pipeline anymore. ABIDE data processed like this did not give good results.
    Go for minimally_preproc_register instead.

    This function copies the original .nii file to the out_dir,
    then calls bbregister on the .nii file found at the root of each subject's folder
    it registers the .nii image to match its freesurfer files

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file
    
    derivative : {"alff","degree_binarize","degree_weighted","dual_regression",
               "eigenvector_binarize","eigenvector_weighted","falff","func_mask",
               "func_mean","func_preproc","lfcd","reho","rois_aal","rois_cc200",
               "rois_cc400","rois_dosenbach160","rois_ez","rois_ho","rois_tt","vmhc"}

    subject_folder : string
        path to the folder where is kept the data for each subject to be registered

    out_data : string
        path to folder where newly registered data should be kept

    change_sub_dir :  boolean, optional, default False
        if activated will force SUBJECTS_DIR to be destination folder for each call to os.system

    contrast : {"t1", "dti", "t2", "bold"}
        used by freesurfer's bbregister, the contrast used on the original image 
        (t1 is White Matter brighter than Grey Matter, the others set Grey Matter as brightest)

    Notes
    -----
    This function will only be called on subjects which have not already been registered, 
    by checking they do not have a "_register" file in out_data.
    """
    cmd_base = ""
    if not os.path.exists(out_data) or not os.path.exists(
        os.path.join(out_data, subject)
    ):
        os.makedirs(os.path.join(out_data, subject))
    if not os.path.exists("{1}/{0}/{0}_register".format(subject, out_data)):
        shutil.copy2(
            "{1}/{0}/{0}_{2}.nii.gz".format(subject, subject_folder, derivative),
            "{1}/{0}/{0}_{2}.nii.gz".format(subject, out_data, derivative),
        )

        if change_sub_dir:
            cmd_base = "export SUBJECTS_DIR=" + subject_folder + "&& "

        cmd = (
            cmd_base
            + "bbregister --s {0} --mov {1}/{0}/{0}_{2}.nii.gz --reg {1}/{0}/{0}_register.dat --{3}".format(
                subject, out_data, derivative, contrast
            )
        )

        os.system(cmd)


def minimally_preproc_register(
    subject, derivative, subject_folder, out_data, change_sub_dir=False, contrast="t1",
):

    """
    This function copies the original .nii file to the out_dir,
    then calls bbregister on the minimally preprocessed .nii file found at the root of each subject's folder
    the matrix used to transformthat .nii image to match its freesurfer files is also applied to the normally
    processed data, making a new file in the intermediary data folder.

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file
    
    derivative : {"alff","degree_binarize","degree_weighted","dual_regression",
               "eigenvector_binarize","eigenvector_weighted","falff","func_mask",
               "func_mean","func_preproc","lfcd","reho","rois_aal","rois_cc200",
               "rois_cc400","rois_dosenbach160","rois_ez","rois_ho","rois_tt","vmhc"}

    subject_folder : string
        path to the folder where is kept the data for each subject to be registered

    out_data : string
        path to folder where newly registered data should be kept

    change_sub_dir :  boolean, optional, default False
        if activated will force SUBJECTS_DIR to be destination folder for each call to os.system

    contrast : {"t1", "dti", "t2", "bold"}
        used by freesurfer's bbregister, the contrast used on the original image 
        (t1 is White Matter brighter than Grey Matter, the others set Grey Matter as brightest)

    Notes
    -----
    This function will only be called on subjects which have not already been registered, 
    by checking they do not have a "_register" file in out_data.
    """
    cmd_base = ""
    if not os.path.exists(out_data) or not os.path.exists(
        os.path.join(out_data, subject)
    ):
        os.makedirs(os.path.join(out_data, subject))
    if not os.path.exists("{1}/{0}/{0}_register.dat".format(subject, out_data)):

        if change_sub_dir:
            cmd_base = "export SUBJECTS_DIR=" + subject_folder + "&& "

        cmd = (
            cmd_base
            + "bbregister --s {0} --mov {3}/{0}/{0}_func_minimal.nii.gz --o {1}/{0}/{0}_func_minimal.nii.gz --reg {1}/{0}/{0}_register.dat --{2}".format(
                subject, out_data, contrast, subject_folder
            )
        )
        os.system(cmd)
        print("register.dat has been calculated on minimally preprocessed data")
        cmd = "mri_vol2vol --reg {1}/{0}/{0}_register.dat --mov {3}/{0}/{0}_{2}.nii.gz --fstarg --o {1}/{0}/{0}_{2}.nii.gz --no-resample".format(
            subject, out_data, derivative, subject_folder
        )
        os.system(cmd)

        print("register.dat has been applied on volume")
    else:
        print(
            "a register.dat file allready exists at {1}/{0}\n if you wish to run again, remove that file.".format(
                subject, out_data
            )
        )


def split_dim_time(subject, derivative, out_data):
    """
    Takes the .nii file found at the root of a given subject's folder in subject_folder
    will split it temporally into as many .nii files as they are time frames.
    These new files will be located in the same folder, in a new directory : '/splitted'

    will only split files for subjects which do not allready have a 'splitted' directory
    in their subject folder, to avoid calling the same function on a subject multiple times.

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    derivative : {"alff","degree_binarize","degree_weighted","dual_regression",
               "eigenvector_binarize","eigenvector_weighted","falff","func_mask",
               "func_mean","func_preproc","lfcd","reho","rois_aal","rois_cc200",
               "rois_cc400","rois_dosenbach160","rois_ez","rois_ho","rois_tt","vmhc"}

    out_data : string
        path to folder where new splitted data will be kept. A folder named splitted will automatically be added to that path
    
    
    """

    destination = "{}/{}/splitted/".format(out_data, subject)
    if not os.path.exists(destination):
        cmd = "fslsplit {}/{}/{}_{}.nii.gz {}_{}_Res".format(
            out_data, subject, subject, derivative, subject, derivative,
        )
        os.system(cmd)

        os.makedirs(destination)
        for file in glob.glob("{}_{}_Res*".format(subject, derivative)):
            shutil.move(file, destination)
        print("{} split done\n".format(subject))
    else:
        print(
            "splitted folder already exists, skipping {} remove folder if you wish to run again \n".format(
                subject
            )
        )


def check_and_project_vol2surf(
    subject, derivative, fs_subdir, out_dir, template="fsaverage5",
):
    """
    Makes a splitted .nii files into splitted .gii files after checking it has not already been done for a given subject

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    derivative : {"alff","degree_binarize","degree_weighted","dual_regression",
               "eigenvector_binarize","eigenvector_weighted","falff","func_mask",
               "func_mean","func_preproc","lfcd","reho","rois_aal","rois_cc200",
               "rois_cc400","rois_dosenbach160","rois_ez","rois_ho","rois_tt","vmhc"}

    fs_subdir : string path to file
        is the path to the folder where subject data will be found

    out_data : string
        path to folder where the projected data will be saved. This must be the same folder where the splitted Nifti files are found
    
    template : string, optional ("fsaverage5" by default)       
        name of the template used bu freesurfer functions, must be found in freesurfer's SUBJECTS_DIR

    
    """
    split_dir = "{}/{}/splitted/".format(out_dir, subject)
    # here we check that no gii have already been built for this subject, that the projection has not allready been attempted
    if len(glob.glob(split_dir + "{}_{}_Res*.gii".format(subject, derivative))) == 0:

        for file in glob.glob(
            split_dir + "{}_{}_Res*.nii.gz".format(subject, derivative)
        ):
            filename = file[len(split_dir) :]

            project_vol2surf(
                fs_subdir,
                subject,
                file,
                filename,
                split_dir,
                tgt_subject=template,
                hem_list=["lh", "rh"],
                sfwhm=0,
            )
            print("projection of {} done\n".format(subject))
    else:
        print(
            "!! Subject {} already has .gii files generated, if you wish to regenerate .gii files, remove all .gii files from {}\n".format(
                subject, split_dir
            )
        )


def project_vol2surf(
    fs_subdir,
    sub,
    nii_file,
    filename,
    gii_dir,
    tgt_subject,
    hem_list=["lh", "rh"],
    sfwhm=0,
):
    """
    Project one Nifti file (3D image) to surface saved as Gifti file.
    Projection is done for left and right
    
    Parameters
    ----------
          : FreeSurfer subjects directory

        sub: Subject name

        nii_file: Splitted .nii directory

        gii_dir: Output directory

        gii_sfx: Gifti files suffix (add to the hemisphere name)

        tgt_subject: Name of target subject

        hem_list: Hemispheres (default: left and right)

        sfwhm: Surface smoothing (default = 0mm)
    """
    for hem in hem_list:
        gii_file = "{}/{}.{}_{}.gii".format(gii_dir, filename, hem, tgt_subject)

        cmd = (
            "$FREESURFER_HOME/bin/mri_vol2surf --src {} --o {} "
            "--out_type gii --regheader {} --hemi {} "
            "--projfrac-avg 0 1 0.1 --surf-fwhm {:d} --sd {} "
            "--trgsubject {}".format(
                nii_file, gii_file, sub, hem, sfwhm, fs_subdir, tgt_subject
            )
        )
        os.system(cmd)


def check_and_correlate(
    subject,
    template="fsaverage5",
    fs_subdir="/scratch/mmahaut/data/abide/downloaded_preprocessed",
    intermediary_dir="./intermediary_data",
    out_dir="./processed_ABIDE",
):
    """
    Makes the correlation matrix between ROIs and Voxels only when this step has not already been done for a given subject

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file  

    template : string, optional ("fsaverage5" by default)       
        name of the template used bu freesurfer functions, must be found in freesurfer's SUBJECTS_DIR  

    fs_subdir : string path to file
        is the path to the folder where subject data will be found

    intermediary_data_path : string, optional ("/scratch/mmahaut/data/abide/intermediary" is default)
        path to the file where all intermediary data will be saved and looked for, included gifti hemispheral surface files

    out_dir : string
        path to folder where the correlation matrix will be saved
    

    """
    split_dir = "{}/{}/splitted/".format(intermediary_dir, subject)
    if not os.path.exists(
        out_dir + "/correlation_matrix_{}_{}.npy".format(template, subject)
    ):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        correlation(fs_subdir, subject, template, split_dir, intermediary_dir, out_dir)
    else:
        print("correlation matrix already exits for {}".format(subject))


def correlation(subdir, sub, template, split_dir, intermediary_dir, out_dir):
    """"
    This code allows to compute the correlation bewteen voxels and ROIs.
    It needs a set of labels (annotation files) and gii files.
    the template file must be in subdir with the subjects raw data
    The code is decomposed into three phases (procedures)
        :proc  1: matrix construction of gii file (each line is a voxel, and the column is the j time serie)
        :proc  2: : for each ROI, we save the set of selected voxels based on the annotation file (labels)
        :proc  3: Coorelation matrix, for each voxel we compute their correlation with the average value of each ROI


    """
    # trgt_fsaverage = "/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/fsaverage/label/"
    # trgt_fsaverage5 = "/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/fsaverage5/label/"
    trgt = "{}/{}/label/".format(subdir, template)

    subname = sub
    start = time.time()

    # FS AVERAGE 5
    """""
    STEP 1: MATRIX CONSTRUCTION  OF lh.Gifti AND rh.Gifti files
    """ ""
    print("STEP 1: MATRIX CONSTRUCTION  OF lh.Gifti AND rh.Gifti files")
    hem_list = ["lh", "rh"]
    for hem in hem_list:
        gii_matrix = np.empty([])

        print("load of " + hem + ".gifti files")
        for load_file in glob.glob(split_dir + "*{}_{}.gii".format(hem, template)):
            filename = load_file
            gii = nib.load(filename)
            data = np.array([gii.darrays[0].data])
            if gii_matrix.shape == ():
                gii_matrix = data
            else:
                gii_matrix = np.concatenate((gii_matrix, data), axis=0)
        gii_matrix = np.transpose(gii_matrix)
        print("Size of Matrix " + hem + ":", gii_matrix.shape)

        save_file = (
            intermediary_dir
            + "/"
            + subname
            + "/glm/noisefiltering/gii_matrix_{}_{}.npy".format(template, hem)
        )

        if not os.path.exists(
            intermediary_dir + "/" + subname + "/glm/noisefiltering/"
        ):
            os.makedirs(intermediary_dir + "/" + subname + "/glm/noisefiltering/")

        np.save(save_file, gii_matrix)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "Elapsed time Step 1: {:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        )
        start = time.time()
        """""
        Step 2 : ROI averaging.
        """ ""
        # Read Annotation LH file
        print("STEP 2: ROI AVERAGING")
        print("Read annotation {} file: {}.aparc.a2009s.annot".format(hem, hem))
        # filepath = subdir + "/" + subname + "/label/{}.aparc.a2009s.annot".format(hem) # espace natif
        filepath = trgt + "{}.aparc.a2009s.annot".format(hem)  # espace fsaverage
        [labels, ctab, names] = nib.freesurfer.io.read_annot(filepath, orig_ids=False)
        print("labels {}".format(hem), labels)
        print("Size of labels", labels.shape)
        # print("List of names", names)
        print("Number of ROIs", len(names))
        # ID Regions Extraction
        print("ID ROI extraction")
        Id_ROI = np.asarray(sorted(set(labels)))
        print("Ids ROIs:", Id_ROI)
        print("ROIs dimensions:", Id_ROI.shape)
        # print("len",len(labels))

        # Extract time series for each ROI (averaging)
        print("Extract time series for each ROI by averaging operation")
        roi_avg = np.empty((len(Id_ROI), gii_matrix.shape[1]))
        for i in range(0, len(Id_ROI)):
            print("*********************************************************")
            print("ID ROI:", Id_ROI[i])
            mask = np.where(labels == Id_ROI[i])
            roi_timeseries = gii_matrix[mask, :].mean(1)
            roi_avg[i, :] = roi_timeseries
        print("*********************************************************")
        print("********** Results: **********")
        print("Size of the Average Matrix of ALL ROIs", roi_avg.shape)
        # Save the average matrix of all ROIs
        if hem == "lh":
            file = (
                intermediary_dir
                + "/"
                + subname
                + "/glm/noisefiltering/roi_avg_lh_{}.npy".format(template)
            )
            np.save(file, roi_avg)
        else:
            file = (
                intermediary_dir
                + "/"
                + subname
                + "/glm/noisefiltering/roi_avg_rh_{}.npy".format(template)
            )
            np.save(file, roi_avg)
        print("*********************************************************")
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "Elapsed time Step 2: {:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        )
        print("*********************************************************")
        start = time.time()
    """""
    Step 3 : Correlation Matrix
    """ ""
    print("STEP 3: COMPUTING OF THE CORRELATION MATRIX ")
    roi_avg_lh = np.load(
        intermediary_dir
        + "/"
        + subname
        + "/glm/noisefiltering/roi_avg_lh_{}.npy".format(template)
    )
    roi_avg_rh = np.load(
        intermediary_dir
        + "/"
        + subname
        + "/glm/noisefiltering/roi_avg_rh_{}.npy".format(template)
    )
    roi_avg = np.concatenate((roi_avg_lh, roi_avg_rh))
    print("roi avg shape", roi_avg.shape)
    gii_matrix_lh = np.load(
        intermediary_dir
        + "/"
        + subname
        + "/glm/noisefiltering/gii_matrix_{}_lh.npy".format(template)
    )
    gii_matrix_rh = np.load(
        intermediary_dir
        + "/"
        + subname
        + "/glm/noisefiltering/gii_matrix_{}_rh.npy".format(template)
    )
    gii_matrix = np.concatenate((gii_matrix_lh, gii_matrix_rh))
    print("gii matrix shape", gii_matrix.shape)
    correlation_matrix = np.empty((gii_matrix.shape[0], roi_avg.shape[0]))
    print("correlation matrix shape", correlation_matrix.shape)
    for n in range(gii_matrix.shape[0]):
        for m in range(roi_avg.shape[0]):
            correlation_matrix[n, m] = pearsonr(gii_matrix[n, :], roi_avg[m, :])[0]
    correlation_matrix[np.where(np.isnan(correlation_matrix[:]))] = 0
    file = out_dir + "/correlation_matrix_{}_{}.npy".format(template, subname)

    np.save(file, correlation_matrix)
    print("********** Results: **********")
    print("Dimensions of the correlation Matrix:", correlation_matrix.shape)
    print("Computing of the correlation matrix, DONE!:", subname)
    test = np.isnan(correlation_matrix[:])
    if True in test[:]:
        print("Nan values exist in correlation matrix")
        print("Number of Nan Values:", np.sum(test))
    else:
        print("No Nan values in correlation matrix")
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        "Elapsed time Step 3: {:0>2}:{:0>2}:{:05.2f}".format(
            int(hours), int(minutes), seconds
        )
    )


def prepare_matlab(
    subject, subject_folder, out_dir,
):
    """
    Makes .mat files out of .white files to be used to get the eigenvectors

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    subject_folder : string path to file
        is the path to the folder where subject data will be found

    out_dir : string
        path to folder where the .mat matrix will be saved. when using the complete pipeline, it should be the intermediary dir.
    """
    hem_list = ["lh", "rh"]
    out_path = "{}/{}/".format(out_dir, subject)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for hem in hem_list:
        geom_in = nib.freesurfer.io.read_geometry(
            "{}/{}/surf/{}.white".format(subject_folder, subject, hem)
        )
        geom_out = np.array((geom_in[0] + 1, geom_in[1] + 1))
        scio.savemat(
            out_path + "{}_{}_white.mat".format(subject, hem), {hem: geom_out},
        )


def matlab_find_eig(subject, white_matlab_matrix, matlab_runtime_path, script_path):
    """
    Calls on spangy to build a map of gyrifications

    Parameters
    ----------

    subject : string, subject name
        subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

    white_matlab_matrix : string
        path to the folder where the hemispheric .white data that has been converted to a .mat file can be found

    matlab_runtime_path : string path to file
         path to the folder where Runtime's bin folder is found, used for the gyrification computing
   
    script_path : string
        path to the folder where the .sh run_find_eig.sh and find_eig are found, used for gyrification computing

    out_dir : string
        path to the file where the gyrification eigen-vector matrix will be saved. If the path does not exist, folders will be added.


    """
    if not os.path.exists(white_matlab_matrix + "/{}_lheig_vec.mat".format(subject)):
        if not os.path.exists(white_matlab_matrix):
            os.makedirs(white_matlab_matrix)

        cmd = "{}/run_find_eig.sh {} {} {} {}".format(
            script_path,
            matlab_runtime_path,
            subject,
            white_matlab_matrix,
            white_matlab_matrix,
        )
        os.system(cmd)
        print("Gyrification matrix written for " + subject)
    else:
        print(
            "A .mat file already exists for {} in {} \n If you wish it to be generated again, you must remove it".format(
                subject, white_matlab_matrix
            )
        )


# def file_movement(out_dir, intermediary_dir):
#     # temporary function here to move files back to the intermediary folder, where they belong.
#     for file in glob.glob(out_dir + "/*.mat"):
#         print(file[(len(out_dir) + 1) : (len(file) - 14)])
#         shutil.move(
#             file,
#             intermediary_dir
#             + "/{}/".format(file[(len(out_dir) + 1) : (len(file) - 14)]),
#         )


# def file_movement2(
#     out_dir="/scratch/mmahaut/data/abide/features_gyrification",
#     intermediary_dir="/scratch/mmahaut/data/abide/intermediary",
# ):
#     # temporary function here to move files back to the intermediary folder, where they belong.
#     for file in glob.glob(out_dir + "/*.npy"):
#         print(file[(len(out_dir) + 1) : (len(file) - 25)])
#         shutil.move(
#             file,
#             intermediary_dir
#             + "/{}/".format(file[(len(out_dir) + 1) : (len(file) - 25)]),
#         )


def align_gyrification(subject, intermediary_dir, template="fsaverage5"):
    """
    Surfaces are all maped on the chosen template surface, giving us neat matrices with matching dimensions !
    Existig matrix is segmented and each eigenvector is projected on fsaverage5.



    Parameters
    ----------
        subject : string, subject name
            subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file

        intermediary_data_path : string, optional ("/scratch/mmahaut/data/abide/intermediary" is default)
            path to the file where all intermediary data will be saved and looked for, included gifti hemispheral surface files

        template : string, optional ("fsaverage5" by default)       
            name of the template used bu freesurfer functions, must be found in freesurfer's SUBJECTS_DIR  


    Notes
    -----
        The input matrices loaded from chosen path are transposed from a (voxel, eigenvector) shape on input to (eigenvector, voxel) on output
        (this makes line operation in for loops more relevant, and is useful for the gyrification_sign() function later in the pipeline)
    """
    hem_list = ["lh", "rh"]
    for hem in hem_list:
        if not os.path.exists(
            "{}/{}/{}_{}eig_vec_{}.npy".format(
                intermediary_dir, subject, subject, hem, template
            )
        ):
            matlab_matrix = scio.loadmat(
                "{}/{}/{}_{}eig_vec.mat".format(intermediary_dir, subject, subject, hem)
            )
            matrix = matlab_matrix["eigVec"]
            matrix = np.transpose(matrix)
            # Transposition is done so that the lines in the for loop correspond to the different eigenvectors.
            # That transposition is not reversed at the end of the function, as this is also relevant during sign checking
            out_matrix = np.empty([])

            # Splitting the gyrification matrix into independent .gii files and saving them
            for i in range(len(matrix)):
                gii_image = nib.gifti.gifti.GiftiImage()
                gii_data_array = nib.gifti.gifti.GiftiDataArray(data=matrix[i])
                gii_image.add_gifti_data_array(gii_data_array)
                gii_file_name = "{}/{}/splitted_gyr/{}_{}_segmented_{}.gii".format(
                    intermediary_dir, subject, subject, hem, i
                )

                if not os.path.exists(
                    "{}/{}/splitted_gyr/".format(intermediary_dir, subject)
                ):
                    os.makedirs("{}/{}/splitted_gyr/".format(intermediary_dir, subject))

                nib.save(gii_image, gii_file_name)

                gii_corrected_file_name = "{}/{}/splitted_gyr/{}_{}_segmented_{}_{}.gii".format(
                    intermediary_dir, subject, subject, hem, template, i
                )

                if not os.path.exists("{}/splitted_gyr/".format(intermediary_dir)):
                    os.makedirs("{}/splitted_gyr/".format(intermediary_dir))

                # calling mri_surf2surf on each of those images to put them in fsaverage5 space
                cmd = "mri_surf2surf --srcsubject {} --srcsurfval {} --trgsubject {} --trgsurfval {} --trgsurfreg sphere.reg --hemi {} --sfmt gifti --tfmt gifti --noreshape --no-cortex".format(
                    subject, gii_file_name, template, gii_corrected_file_name, hem
                )
                os.system(cmd)
                # putting the segmented matrix back together
                if out_matrix.shape == ():
                    out_matrix = np.array(
                        [nib.load(gii_corrected_file_name).darrays[0].data]
                    )
                else:
                    out_matrix = np.concatenate(
                        (
                            out_matrix,
                            np.array(
                                [nib.load(gii_corrected_file_name).darrays[0].data]
                            ),
                        ),
                        axis=0,
                    )

            np.save(
                "{}/{}/{}_{}eig_vec_{}.npy".format(
                    intermediary_dir, subject, subject, hem, template
                ),
                out_matrix,
            )
        else:
            print(
                "{}/{}/{}_{}eig_vec_{}.npy already exists. Remove from file to run again".format(
                    intermediary_dir, subject, subject, hem, template
                )
            )


def gyrification_sign(
    subject, ref_subject, out_dir, intermediary_dir, template="fsaverage5"
):
    """
    Make eigen-vector signs coherent throughout all corresponding eigenvectors by matching them with a given reference subject.

    Parameters
    ----------
        subject : string, subject name
            subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file
        
        ref_subject : string, reference subject name
            subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file, all eigenvector sign will match

        out_dir : string
            path to the file where the gyrification eigen-vector matrix will be saved. If the path does not exist, folders will be added.
        
        intermediary_data_path : string, optional ("/scratch/mmahaut/data/abide/intermediary" is default)
            path to the file where all intermediary data will be saved and looked for, included gifti hemispheral surface files

        template : string, optional ("fsaverage5" by default)       
            name of the template used bu freesurfer functions, must be found in freesurfer's SUBJECTS_DIR  
    
    Notes
    -----
        The input matrix that is loaded from given path will have a (eigenvector, voxel) shape, but will be transposed to 
        (voxel, eigenvector) shape so as to match the format of the other modalities when entering the auto-encoder

    """
    # we select a subject as the reference subject
    if not os.path.exists(
        "{}/{}_eig_vec_{}_onref_{}.npy".format(out_dir, subject, template, ref_subject)
    ):
        if subject == ref_subject:
            lh_gyr = np.load(
                "{}/{}/{}_{}eig_vec_{}.npy".format(
                    intermediary_dir, subject, subject, "lh", template
                )
            )
            rh_gyr = np.load(
                "{}/{}/{}_{}eig_vec_{}.npy".format(
                    intermediary_dir, subject, subject, "rh", template
                )
            )
        else:
            lh_gyr = hemisphere_sign_check(
                subject, ref_subject, out_dir, intermediary_dir, "lh", template
            )
            rh_gyr = hemisphere_sign_check(
                subject, ref_subject, out_dir, intermediary_dir, "rh", template
            )

        full_brain_gyr = np.concatenate((lh_gyr.transpose(), rh_gyr.transpose()))
        # !! TODO : The names are begining to be long. It could be interesting to have either a metadata file,
        #  or a text file with the information saved in folder root
        np.save(
            "{}/{}_eig_vec_{}_onref_{}.npy".format(
                out_dir, subject, template, ref_subject
            ),
            full_brain_gyr,
        )
        print("sign check done for {}".format(subject))
    else:
        print(
            "{}/{}_eig_vec_{}_onref_{}.npy file already exists. If you wish to run agein, remove file from directory".format(
                out_dir, subject, template, ref_subject
            )
        )


def hemisphere_sign_check(
    subject, ref_subject, out_dir, intermediary_dir, hem, template="fsaverage5"
):
    """
    Make eigen-vector signs coherent throughout all corresponding eigenvectors by matching them with a given reference subject.

    Parameters
    ----------
        subject : string, subject name
            subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file
        
        ref_subject : string, reference subject name
            subject name to call function on, as found in freesurfer's SUBJECTS_DIR or in the subs_list file, all eigenvector sign will match

        out_dir : string
            path to the file where the gyrification eigen-vector matrix will be saved. If the path does not exist, folders will be added.
        
        intermediary_data_path : string, optional ("/scratch/mmahaut/data/abide/intermediary" is default)
            path to the file where all intermediary data will be saved and looked for, included gifti hemispheral surface files

        template : string, optional ("fsaverage5" by default)       
            name of the template used bu freesurfer functions, must be found in freesurfer's SUBJECTS_DIR  
    
    Notes
    -----
        The input matrix that is loaded from given path will have a (eigenvector, voxel) shape, but will be transposed to 
        (voxel, eigenvector) shape so as to match the format of the other modalities when entering the auto-encoder

    """
    ref_gyr_mat = np.load(
        "{}/{}/{}_{}eig_vec_{}.npy".format(
            intermediary_dir, ref_subject, ref_subject, hem, template
        )
    )
    gyr_mat = np.load(
        "{}/{}/{}_{}eig_vec_{}.npy".format(
            intermediary_dir, subject, subject, hem, template
        )
    )

    for i in range(len(gyr_mat)):
        # Le signe du produit scalaire nous indique si les deux vecteurs propres ont le même signe
        prod_scal = np.vdot(gyr_mat[i], ref_gyr_mat[i])
        if prod_scal < 0:
            gyr_mat[i] = -1 * gyr_mat[i]

    # Transposition is required to correspond to other modality formats ie (voxel, feature)
    return gyr_mat


########
# Launch
########
if __name__ == "__main__":
    # execute only if run as a script
    data_list_files = "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/files_to_download.json"
    data_list_file = open(data_list_files)
    data_list = json.load(data_list_file)

    if data_list["data"] == "ABIDE":
        extract_one_abide(
            sys.argv[1],
            sys.argv[2],
            data_list,
            raw_data_path="/scratch/mmahaut/data/abide/downloaded_preprocessed",
            force_destination_folder=False,
            template="fsaverage5",
            contrast="t1",
            rfmri_features_data_path="/scratch/mmahaut/data/abide/features_rsfMRI",
            gyrification_features_data_path="/scratch/mmahaut/data/abide/features_gyrification",
            matlab_runtime_path="/scratch/mmahaut/tools/MATLAB_Runtime/v95",
            matlab_script_path="/scratch/mmahaut/scripts/INT_fMRI_processing/for_redistribution_files_only",
            intermediary_data_path="/scratch/mmahaut/data/abide/intermediary",
        )
    elif data_list["data"] == "interTVA":
        extract_one_interTVA(
            sys.argv[1],
            sys.argv[2],
            data_list,
            raw_data_path="/scratch/mmahaut/data/intertva/downloaded_preprocessed",
            template="fsaverage5",
            gyrification_features_data_path="/scratch/mmahaut/data/intertva/features_gyrification",
            matlab_runtime_path="/scratch/mmahaut/tools/MATLAB_Runtime/v95",
            matlab_script_path="/scratch/mmahaut/scripts/INT_fMRI_processing/for_redistribution_files_only",
            intermediary_data_path="/scratch/mmahaut/data/intertva/intermediary",
        )

