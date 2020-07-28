# import keras
# from keras.layers import Input, Dense, concatenate
# from keras.models import Model
# from keras.layers import Dropout
# from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import sys
import os
import json

# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# from keras.optimizers import SGD, Adadelta, Adam
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

# from keras.callbacks import ModelCheckpoint
# from keras.models import load_model
# import keras.backend as K


def run_slurm_job_mdae(
    data_orig,
    data_type,
    dimension,
    fold,
    email="mmahaut@ensc.fr",
    logs_dir="/scratch/mmahaut/scripts/logs",
    python_path="python",
    slurm_dir="/scratch/mmahaut/scripts/slurm",
    code_dir="/scratch/mmahaut/scripts/INT_fMRI_processing",
    script_name="mdae_step.py",
):

    # subs_list_file = open(subs_list_file_path)
    # subject_list = json.load(subs_list_file)
    # # An arbitrary reference subject has to be chosen. Here we just take the first.
    # ref_subject = subject_list[0]

    job_name = "{}_dim{}_fold{}_mdae".format(data_orig, dimension, fold)
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
        fh.writelines("#SBATCH --partition=kepler\n")
        fh.writelines("#SBATCH --gres-flags=enforce-binding\n")
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
            + "conda activate tf_gpu\n"
            + "{} {}/{} {} {} {} {}".format(
                python_path,
                code_dir,
                script_name,
                data_orig,
                data_type,
                dimension,
                fold,
            )
        )
        fh.writelines(batch_cmd)

    os.system("sbatch %s" % slurmjob_path)


if __name__ == "__main__":

    data_orig = sys.argv[1]
    # sys.argv[2] Default is gyrification, tfMRI must be written otherwise !!check what happens when empty
    data_type = sys.argv[2]

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

    dimensions = [20]

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

        index_subjects = np.arange(0, len(sub_list))
        for dim in dimensions:
            fold = 0
            for train_index, test_index in kf.split(index_subjects, Y):
                fold += 1
                fold_path = "/scratch/mmahaut/data/abide/ae_gyrification/{}/fold_{}".format(
                    dim, fold
                )
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)
                np.save(os.path.join(fold_path, "train_index.npy"), train_index)
                np.save(os.path.join(fold_path, "test_index.npy"), test_index)
                run_slurm_job_mdae(data_orig, data_type, dim, fold)

    elif data_orig == "interTVA":
        kf = KFold(n_splits=10)

        sub_list_files = "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list.json"
        sub_list_file = open(sub_list_files)
        sub_list = json.load(sub_list_file)

        index_subjects = np.arange(0, len(sub_list))
        for dim in dimensions:
            fold = 0

            for train_index, test_index in kf.split(index_subjects):
                fold += 1
                ae_type = "ae" if data_type == "tfMRI" else "ae_gyrification"
                fold_path = "/scratch/mmahaut/data/intertva/{}/{}/fold_{}".format(
                    ae_type, dim, fold
                )
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)
                np.save(os.path.join(fold_path, "train_index.npy"), train_index)
                np.save(os.path.join(fold_path, "test_index.npy"), test_index)
                run_slurm_job_mdae(data_orig, data_type, dim, fold)
    else:
        print(
            "Warning !! : Please provide data origin as parameter when calling script: either 'ABIDE' or 'interTVA' "
        )

