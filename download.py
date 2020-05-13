import os
import json


def download_abide_urls(
    subject_list="./url_preparation/subs_list.json",
    data_list_files="./url_preparation/files_to_download.json",
    destination_folder="./rsfMRI_ABIDE",
):
    """
    Here we build the urls as described by ABIDE documentation here : 
    http://preprocessed-connectomes-project.org/abide/download.html

    we expect to find in the data_list_files the PATH to a .JSON, which determines
        1 : the folowing parameters, once, for the resting state fMRI file of all subjects :

            [pipeline] = ccs | cpac | dparsf | niak 
            [strategy] = filt_global | filt_noglobal | nofilt_global | nofilt_noglobal
            [file identifier] = the FILE_ID value from the summary spreadsheet
            [derivative] = alff | degree_binarize | degree_weighted | dual_regression | ... 
                eigenvector_binarize | eigenvector_weighted | falff | func_mask | ... 
                func_mean | func_preproc | lfcd | reho | rois_aal | rois_cc200 | ... 
                rois_cc400 | rois_dosenbach160 | rois_ez | rois_ho | rois_tt | vmhc

        2: which files to download in the freesurfer file tree
    
    each file is then aquired and put in the right folder using wget

    This function requires the subs_list.json, listing which subjects to consider amongst those found in the .yml here :
    https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/yamls/subs_list.yml

    TODO :  only download files which have not been downloaded in the past
            add user feedback
            mkdir only works on LINUX for now
    """
    # opening .json
    subs_list_file = open(subject_list)
    subs_list = json.load(subs_list_file)
    data_list_file = open(data_list_files)
    data_list = json.load(data_list_file)

    for i in range(len(subs_list)):

        # Adding rsFMRI file
        cmd = (
            "wget -q -P {}/{} https://s3.amazonaws.com/fcp-indi/data/Projects/"
            + "ABIDE_Initiative/Outputs/{}/{}/{}/{}_{}.nii.gz "
        ).format(
            destination_folder,
            subs_list[i],
            data_list["rsfMRI"]["pipeline"],
            data_list["rsfMRI"]["strategy"],
            data_list["rsfMRI"]["derivative"],
            subs_list[i],
            data_list["rsfMRI"]["derivative"],
        )
        os.system(cmd)

        # Adding freesurfer directory
        for key in data_list["freesurfer"]:
            os.system(
                "mkdir -p {}/{}/{}".format(destination_folder, subs_list[i], key)
            )  # only works on LINUX
            for file in data_list["freesurfer"][key]:
                cmd = (
                    "wget -q -P {}/{}/{} https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/"
                    + "{}/{}/{} "
                ).format(destination_folder, subs_list[i], key, subs_list[i], key, file)
                os.system(cmd)
