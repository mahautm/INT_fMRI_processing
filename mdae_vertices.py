import matplotlib.pyplot as plt

plt.switch_backend("agg")
import sys
import os
import json
import errno

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from mdae import run_slurm_job_mdae

if __name__ == "__main__":

    data_orig = sys.argv[1]  # ABIDE or interTVA
    data_type = sys.argv[2]  # tfMRI or gyrification

    # IJCNN paper points to 20 being the best dimension, with 5 to rsfMRI and 15 to tfMRI
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
                    run_slurm_job_mdae(
                        data_orig,
                        data_type,
                        "",
                        "15-5_vertex",
                        fold,
                        script_name="mdae_step_vertices.py",
                    )

    elif data_orig == "interTVA":
        kf = KFold(n_splits=10)

        sub_list_files = "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list.json"
        sub_list_file = open(sub_list_files)
        sub_list = json.load(sub_list_file)

        index_subjects = np.arange(0, len(sub_list))
        index_vertices = np.arange(
            0, 20484
        )  # <-- number of vertices for a given subject

        index_subject_vertices = np.array(
            np.meshgrid(index_subjects, index_vertices)
        ).T.reshape(
            -1, 2
        )  # <-- all combinations of vertices and subjects

        for dim_1 in dimensions_1:
            for dim_2 in dimensions_2:
                fold = 0

                for train_index, test_index in kf.split(index_subject_vertices):
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
                    run_slurm_job_mdae(
                        data_orig,
                        data_type,
                        "",
                        "15-5_vertex",
                        fold,
                        script_name="mdae_step_vertices.py",
                    )
    else:
        print(
            "Warning !! : Please provide data origin as parameter when calling script: either 'ABIDE' or 'interTVA' "
        )

