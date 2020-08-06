import matplotlib.pyplot as plt

plt.switch_backend("agg")
import sys
import os
import json
import errno

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def run_slurm_job_mdae(
    data_orig,
    data_type,
    dimension_1,
    dimension_2,
    fold,
    email="mmahaut@ensc.fr",
    logs_dir="/scratch/mmahaut/scripts/logs",
    python_path="python",
    slurm_dir="/scratch/mmahaut/scripts/slurm",
    code_dir="/scratch/mmahaut/scripts/INT_fMRI_processing",
    script_name="mdae_step_vertices.py",
):
    """
    This will write a .sh file to train a multi-modal auto-encoder on a specific set of subjects with SLURM, and then execute it.

    Parameters
    ----------
    data_orig : {"ABIDE","interTVA"}
        indicates which data set is used. ABIDE is a dataset with subjects on the autism spectrum and control subjects,
        InterTVA is a dataset where non-pathological subjects are given sound recognition tasks. 

    data_type : {"tfMRI","gyrification"}
        The multi-modal auto-encoder uses two modalities to build it's representations. One is resting-state fMRI, and the other
        is either task fMRI (tfMRI) or an anatomical modality which represents the folds in the subject's brain, (gyrification)

    dimension_1 : int
        The number of encoding layer dimensions for the first modality,
        According to IJCNN 2020 published paper, for tfMRI 15 is the best value

    dimension_2 : int
        The number of encoding layer dimensions for the first modality,
        According to IJCNN 2020 published paper, for rsfMRI 5 is the best value

    fold : int
        For cross-validation, subjects are separated in a train and a test group. Those groups must be saved in the corresponding folder
        the number corresponding to each fold will here be used to load the proper group in the mdae-step.py script

    email : string, email address, default "mmahaut@ensc.fr"
        email to which notifications will be sent at the end of the execution of the script for the first fold of a given dimension

    logs_dir : string, path, default "/scratch/mmahaut/scripts/logs"
        path to where both the output and the error logs will be saved.

    python_path : string, path, default "python"
        path to where the python version used to run the mdae_step.py script is found. 3.7 & 3.8 work.
    
    slurm_dir : string, path, default "/scratch/mmahaut/scripts/slurm"
        path to where the .sh script used to enter the slurm queue will be saved before being executed

    code_dir : string, path, default "/scratch/mmahaut/scripts/INT_fMRI_processing"
        path to where the script can be found

    script_name : string, path, default "mdae_step.py"
        script file name
    """

    job_name = "{}_dim{}-{}_fold{}_mdae".format(
        data_orig, dimension_1, dimension_2, fold
    )
    slurmjob_path = os.path.join(slurm_dir, "{}.sh".format(job_name))
    create_slurmjob_cmd = "touch {}".format(slurmjob_path)
    os.system(create_slurmjob_cmd)

    # write arguments into the slurmjob file
    with open(slurmjob_path, "w") as fh:
        fh.writelines("#!/bin/sh\n")
        fh.writelines("#SBATCH --job-name={}\n".format(job_name))
        fh.writelines("#SBATCH -o {}/{}_%j.out\n".format(logs_dir, job_name))
        fh.writelines("#SBATCH -e {}/{}_%j.err\n".format(logs_dir, job_name))
        fh.writelines("#SBATCH --time=3:00:00\n")
        fh.writelines("#SBATCH --account=b125\n")
        fh.writelines("#SBATCH --partition=skylake\n")
        # fh.writelines("#SBATCH --gres-flags=enforce-binding\n")
        # number of nodes for this job
        fh.writelines("#SBATCH --nodes=1\n")
        # number of cores for this job
        fh.writelines("#SBATCH --ntasks-per-node=10\n")  # ??
        # email alerts
        if fold == 1:
            fh.writelines("#SBATCH --mail-type=END\n")
        fh.writelines("#SBATCH --mail-user={}\n".format(email))
        # making sure group is ok for data sharing within group
        batch_cmd = (
            'eval "$(/scratch/mmahaut/tools/Anaconda3/bin/conda shell.bash hook)"\n'
            + "conda activate tf\n"
            + "{} {}/{} {} {} {} {} {}".format(
                python_path,
                code_dir,
                script_name,
                data_orig,
                data_type,
                dimension_1,
                dimension_2,
                fold,
            )
        )
        fh.writelines(batch_cmd)

    os.system("sbatch %s" % slurmjob_path)


if __name__ == "__main__":

    data_orig = sys.argv[1]
    data_type = sys.argv[2]  # tfMRI or gyrification

    # dimensions = [
    #     1,
    #     3,
    #     5,
    #     8,
    #     10,
    #     13,
    #     15,
    #     18,
    #     20,
    #     23,
    #     25,
    #     28,
    #     30,
    #     33,
    #     35,
    #     38,
    #     40,
    #     42,
    #     45,
    #     48,
    #     50,
    # ]

    # IJCNN paper points to 20 being the best dimension, with 5 to rsfMRI and 15 to tfMRI

    # dimensions = [20] # Legacy... it used to give equal importance to both
    dimensions_1 = [15]
    dimensions_2 = [5]

    # In the ABIDE case, we need to get the Y data to ensure proper repartition of asd and non-asd subjects
    Y = []
    if data_orig == "ABIDE":
        sub_file = "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list_asd_classified.json"
        classified_file = open(sub_file)
        classified_dict = json.load(classified_file)
        # no normalisation step (which kind of seems legit for classification)
        for key in classified_dict:
            Y.append([1] if classified_dict[key] == "asd" else [0])

        kf = StratifiedKFold(n_splits=10)

        sub_list_files = "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list_asd.json"
        sub_list_file = open(sub_list_files)
        sub_list = json.load(sub_list_file)

        # VERTICES KFOLD

        index_subjects = np.arange(0, len(sub_list))
        index_vertices = np.arange(
            0, 20484
        )  # <-- number of vertices for a given subject

        index_subject_vertices = np.array(
            np.meshgrid(index_subjects, index_vertices)
        ).T.reshape(
            -1, 2
        )  # <-- all combinations of vertices and subjects

        Y_vertex = np.concatenate(
            [[y] * 20484 for y in Y]
        )  # index_subject_vertex is indexed with subjects first

        for dim_1 in dimensions_1:
            for dim_2 in dimensions_2:
                fold = 0
                for train_index, test_index in kf.split(
                    index_subject_vertices, Y_vertex
                ):
                    fold += 1
                    fold_path = "/scratch/mmahaut/data/abide/ae_gyrification/{}-{}/fold_{}".format(
                        "", "15-5_vertex", fold
                    )
                    if not os.path.exists(fold_path):
                        try:
                            os.makedirs(fold_path)
                        except OSError as exc:
                            if exc.errno != errno.EEXIST:
                                raise
                            pass
                    np.save(os.path.join(fold_path, "train_index.npy"), train_index)
                    np.save(os.path.join(fold_path, "test_index.npy"), test_index)
                    run_slurm_job_mdae(data_orig, data_type, "", "15-5_vertex", fold)

    elif data_orig == "interTVA":
        kf = KFold(n_splits=10)

        sub_list_files = "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list.json"
        sub_list_file = open(sub_list_files)
        sub_list = json.load(sub_list_file)

        index_subjects = np.arange(0, len(sub_list))
        for dim_1 in dimensions_1:
            for dim_2 in dimensions_2:
                fold = 0

                for train_index, test_index in kf.split(index_subjects):
                    fold += 1
                    ae_type = "ae" if data_type == "tfMRI" else "ae_gyrification"
                    fold_path = "/scratch/mmahaut/data/intertva/{}/{}-{}/fold_{}".format(
                        ae_type, "", "15-5_vertex", fold
                    )
                    if not os.path.exists(fold_path):
                        try:
                            os.makedirs(fold_path)
                        except OSError as exc:
                            if exc.errno != errno.EEXIST:
                                raise
                            pass
                    np.save(os.path.join(fold_path, "train_index.npy"), train_index)
                    np.save(os.path.join(fold_path, "test_index.npy"), test_index)
                    run_slurm_job_mdae(data_orig, data_type, "", "15-5_vertex", fold)
    else:
        print(
            "Warning !! : Please provide data origin as parameter when calling script: either 'ABIDE' or 'interTVA' "
        )

