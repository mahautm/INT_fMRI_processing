import os
import json


def generate_abide_urls(
    subject_list="./url_preparation/subs_list.json",
    freesurfer_files="./url_preparation/freesurfer.json",
):
    """
    Here we build the urls as described by ABIDE documentation here : 
    http://preprocessed-connectomes-project.org/abide/download.html

    We save urls in a .txt so they point to : 

        rsfmri with : 
            the cpac pipeline,
            the filt_global strategy,
            and the alff derivative.
        
        the freesurfer subject file

    This function requires the subs_list.json, built from the .yml here :
    https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/yamls/subs_list.yml

    TODO : improve by providing some choice on the subjects to download
    """
    # opening .json
    subs_list_file = open(subject_list)
    subs_list = json.load(subs_list_file)
    freesurfer_file = open(freesurfer_files)
    freesurfer = json.load(freesurfer_file)

    url_list = ""
    for i in range(len(subs_list)):
        # Adding rsFMRI file
        url_list += (
            "https://s3.amazonaws.com/fcp-indi/data/Projects/"
            + "ABIDE_Initiative/Outputs/cpac/filt_global/alff/{}_alff.nii.gz \n".format(
                subs_list[i]
            )
        )

        # Adding freesurfer directory
        for key in freesurfer["subject"]:
            for file in freesurfer["subject"][key]:
                url_list += (
                    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/"
                    + "{}/{}/{} \n".format(subs_list[i], key, file)
                )

    with open("./url_preparation/rsFMRI_url_list.txt", "w") as outfile:
        outfile.write(url_list)


def download_abide_urls(
    url_text_file="./url_preparation/rsFMRI_url_list.txt",
    destination_folder="./rsfMRI_ABIDE",
):
    """
    This function calls wget to access all rsFMRI data from the text file listing their access URLs
    The required text file is by default rsFMRI_url_list.txt 
    it has been built with the generate_abide_urls function


    TODO : improve by allowing parallel downloads
    """
    cmd = "wget -P {} -i {}".format(destination_folder, url_text_file)
    os.system(cmd)


def project_abide(parameter_list):
    pass
