# TODO : Better Comments

import os
import json
import shutil
import glob
import nibabel as nib
import nibabel.gifti as ng
import scipy.io as scio
import numpy as np
import time
from scipy.stats import pearsonr


def extract_all_ABIDE(
    subject_list="./url_preparation/subs_list.json",
    data_list_files="./url_preparation/files_to_download.json",
    destination_folder="./rsfMRI_ABIDE",
    force_destination_folder=False,
    template="fsaverage5",
):
    """
    calls all functions, in order, on all subjects.
    will get all required subject data from json files, 
    and build the correlation matrices with all required feature extraction steps.
    """
    # opening .json
    subs_list_file = open(subject_list)
    subs_list = json.load(subs_list_file)
    data_list_file = open(data_list_files)
    data_list = json.load(data_list_file)
    for subject in subs_list:
        download_abide_urls(subject, data_list, destination_folder)
        split(subject, data_list["rsfMRI"]["derivative"], destination_folder)
        register(
            subject,
            data_list["rsfMRI"]["derivative"],
            force_destination_folder,
            destination_folder,
        )
        project(
            subject, data_list["rsfMRI"]["derivative"], template, destination_folder
        )
        check_and_correlate(template, destination_folder, subject)
        prepare_matlab(subject, destination_folder)

    subs_list_file.close()
    data_list_file.close()


def download_abide_urls(
    subject, data_list, destination_folder="./rsfMRI_ABIDE",
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


def split(
    subject, derivative, subject_folder="./rsfMRI_ABIDE",
):
    """
    will only split files for subjects which do not allready have a 'splited' directory
    in their subject folder.

    """

    destination = "{}/{}/splitted/".format(subject_folder, subject)
    if not os.path.exists(destination):
        cmd = "fslsplit {}/{}/{}_{}.nii.gz {}_{}_Res".format(
            subject_folder, subject, subject, derivative, subject, derivative,
        )
        os.system(cmd)

        os.makedirs(destination)
        for file in glob.glob("{}_{}_Res*".format(subject, derivative)):
            shutil.move(file, destination)
        print("{} split done\n".format(subject))
    else:
        print(
            "splitted folder already exists, skipping {} remove floder if you wish to run again \n".format(
                subject
            )
        )


def register(
    subject,
    derivative,
    change_sub_dir=False,
    subject_folder="./rsfMRI_ABIDE",
    contrast="t1",
):
    """
    contrast can be either bold, dti, t2 or t1
    this function will only be called on subjects which have not allready been registered, 
    by checking they do not have a "_register" file.
    """
    cmd_base = ""
    if change_sub_dir:
        cmd_base = "export SUBJECTS_DIR=" + subject_folder + "&& "

    if not os.path.exists("{1}/{0}/{0}_register".format(subject, subject_folder)):
        cmd = (
            cmd_base
            + "bbregister --s {0} --mov {1}/{0}/{0}_{2}.nii.gz --reg {1}/{0}/{0}_register --{3} --init-spm".format(
                subject, subject_folder, derivative, contrast
            )
        )

        os.system(cmd)


def project(
    subject,
    derivative,
    template="fsaverage5",
    fs_subdir="./rsfMRI_ABIDE",
    data_list_files="./url_preparation/files_to_download.json",
):
    """

    """
    split_dir = "{}/{}/splitted/".format(fs_subdir, subject)
    # here we check that no gii have already been built for this subject, that the projection has not allready been attempted
    if len(glob.glob(split_dir + "{}_{}_Res*.gii".format(subject, derivative))) == 0:

        for file in glob.glob(
            split_dir + "{}_{}_Res*.nii.gz".format(subject, derivative)
        ):
            print("in")
            filename = file[len(split_dir) :]

            project_epi(
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


def project_epi(
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
        :param fs_subdir: FreeSurfer subjects directory
        :param sub: Subject name
        :param nii_file: Splitted .nii directory
        :param gii_dir: Output directory
        :param gii_sfx: Gifti files suffix (add to the hemisphere name)
        :param tgt_subject: Name of target subject
        :param hem_list: Hemispheres (default: left and right)
        :param sfwhm: Surface smoothing (default = 0mm)
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
    subject, template="fsaverage5", fs_subdir="./rsfMRI_ABIDE",
):
    """

    """
    split_dir = "{}/{}/splitted/".format(fs_subdir, subject)
    if not os.path.exists(
        fs_subdir
        + "/"
        + subject
        + "/glm/noisefiltering/correlation_matrix_{}.npy".format(template)
    ):
        correlation(fs_subdir, subject, template, split_dir)
    else:
        print("correlation matrix already exits for {}".format(subject))


def correlation(subdir, sub, template, split_dir):
    """"
    //!! WIP still adapting to ABIDE
    This code allows to compute the correlation bewteen voxels and ROIs.
    It needs a set of labels (annotation files) and gii files.
    The code is decomposed into three phases (procedures)
        :proc  1: matrix construction of gii file (each line is a voxel, and the column is the j time serie)
        :proc  2: : for each ROI, we save the set of selected voxels based on the annotation file (labels)
        :proc  3: Coorelation matrix, for each voxel we compute their correlation with the average value of each ROI

    """
    # trgt_fsaverage = "/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/fsaverage/label/"
    # trgt_fsaverage5 = "/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/fsaverage5/label/"
    trgt = "/usr/local/freesurfer/6.0.0-1/subjects/{}/label/".format(
        template
    )  # this all needs to be in a .json so we can deal with it with more ease

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
            gii = ng.read(filename)
            data = np.array([gii.darrays[0].data])
            if gii_matrix.shape == ():
                gii_matrix = data
            else:
                gii_matrix = np.concatenate((gii_matrix, data), axis=0)
        gii_matrix = np.transpose(gii_matrix)
        print("Size of Matrix " + hem + ":", gii_matrix.shape)

        save_file = (
            subdir
            + "/"
            + subname
            + "/glm/noisefiltering/gii_matrix_{}_{}.npy".format(template, hem)
        )

        if not os.path.exists(subdir + "/" + subname + "/glm/noisefiltering/"):
            os.makedirs(subdir + "/" + subname + "/glm/noisefiltering/")

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
                subdir
                + "/"
                + subname
                + "/glm/noisefiltering/roi_avg_lh_{}.npy".format(template)
            )
            np.save(file, roi_avg)
        else:
            file = (
                subdir
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
        subdir
        + "/"
        + subname
        + "/glm/noisefiltering/roi_avg_lh_{}.npy".format(template)
    )
    roi_avg_rh = np.load(
        subdir
        + "/"
        + subname
        + "/glm/noisefiltering/roi_avg_rh_{}.npy".format(template)
    )
    roi_avg = np.concatenate((roi_avg_lh, roi_avg_rh))
    print("roi avg shape", roi_avg.shape)
    gii_matrix_lh = np.load(
        subdir
        + "/"
        + subname
        + "/glm/noisefiltering/gii_matrix_{}_lh.npy".format(template)
    )
    gii_matrix_rh = np.load(
        subdir
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
    file = (
        subdir
        + "/"
        + subname
        + "/glm/noisefiltering/correlation_matrix_{}.npy".format(template)
    )
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


def prepare_matlab(subject, subject_folder="./rsfMRI_ABIDE"):
    hem_list = ["lh", "rh"]
    for hem in hem_list:
        geom_in = nib.freesurfer.io.read_geometry(
            "{}/{}/surf/{}.white".format(subject_folder, subject, hem)
        )
        geom_out = np.array((geom_in[0] + 1, geom_in[1] + 1))
        scio.savemat(
            "{}/{}/surf/{}_{}_white.mat".format(subject_folder, subject, subject, hem),
            {hem: geom_out},
        )
