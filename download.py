import os
import json


def generate_abide_urls():
    """
    Here we build the urls as described by ABIDE documentation here : 
    http://preprocessed-connectomes-project.org/abide/download.html

    We save urls in a .txt so they point to resting state fMRI with : 
    the cpac pipeline,
    the filt_global strategy,
    and the alff derivative.

    This function requires the subs_list.json, built from the .yml here :
    https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/yamls/subs_list.yml

    TODO : improve by providing some choice on the subjects to download
    """
    subs_list_file = open("subs_list.json")
    subs_list = json.load(subs_list_file)
    # cmd = "wget -P ./rsfMRI_ABIDE "
    url_list = ""
    for i in range(len(subs_list)):
        url_list += (
            "https://s3.amazonaws.com/fcp-indi/data/Projects/"
            + "ABIDE_Initiative/Outputs/cpac/filt_global/alff/{}_alff.nii.gz \n".format(
                subs_list[i]
            )
        )
    with open("rsFMRI_url_list.txt", "w") as outfile:
        outfile.write(url_list)


def download_abide_urls(
    url_text_file="rsFMRI_url_list.txt", destination_folder="./rsfMRI_ABIDE"
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
