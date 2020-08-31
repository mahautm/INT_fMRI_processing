# This script has to be run AFTER mdae_stats.py, it will summarise the evaluation data in a table and load it as a png
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


def build_stat_table(dimensions, orig_path, stat_files, title=""):
    """
    Builds a table from given statistical files of differet multimodal auto-encoders

    Parameters
    ----------
    dimensions : string,
        used to build the path, should be the same as the name of the dimension file used to save,
        could be a number, two numbers seperated by a hyphon, or a word (for 15-5_vertex for example)
    orig_path : list or table of strings
        The root path of the mdae saving folder, for a given modality or data source
    stat_files :
        Which data files to put in the table
    title :
    Title to use for the table

    Output
    ------
    pyplot table with as rows the different stats from the stat_files as rows, 
    and the diferent mdae from orig_path as lines
    """
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    cell_text = []
    rows = []
    columns = []

    for dim_index in range(len(dimensions)):
        rows.append("{} dim encoding".format(dimensions[dim_index]))
        row_cell_text = []
        for filename in stat_files:
            file_path = os.path.join(orig_path, filename)
            stat = np.NaN
            if os.path.exists(file_path):
                stat = np.load(file_path)[0]
            else:
                print("file does not exist : " + file_path)
            row_cell_text.append(stat)
            columns.append(
                filename[: len(filename) - 4]
            )  # enlever le .npy à la fin des noms de fichier
        cell_text.append(row_cell_text)
    the_table = plt.table(
        cellText=cell_text, rowLabels=rows, colLabels=columns, loc="center",
    )
    the_table.scale(4, 2.5)
    plt.draw()
    plt.title(title)
    plt.savefig(
        os.path.join(orig_path, "mse_summary.png"),
        dpi=fig.dpi,
        bbox_inches="tight",
        pad_inches=0.5,
    )
    return the_table


def regroup_stats(base_path, dimensions, nb_folds):
    """
    """
    # Tables:
    mse_train = np.array([])
    mse_test = np.array([])
    # RMSE (gyr+ rsfmri)
    rmse_train = np.array([])
    rmse_test = np.array([])
    #
    # Standard deviation MSE (gyr+ rsfmri)
    std_mse_train = np.array([])
    std_mse_test = np.array([])
    # Standard deviation RMSE (gyr+ rsfmri)
    std_rmse_train = np.array([])
    std_rmse_test = np.array([])
    # MSE (gyr)
    mse_gyr_train = np.array([])
    mse_gyr_test = np.array([])
    # RMSE (gyr)
    rmse_gyr_train = np.array([])
    rmse_gyr_test = np.array([])
    # std mse (gyr)
    std_mse_gyr_train = np.array([])
    std_mse_gyr_test = np.array([])
    # std rmse (gyr)
    std_rmse_gyr_train = np.array([])
    std_rmse_gyr_test = np.array([])

    # MSE (rsfmri)
    mse_rsfmri_train = np.array([])
    mse_rsfmri_test = np.array([])
    # RMSE (rsfmri)
    rmse_rsfmri_train = np.array([])
    rmse_rsfmri_test = np.array([])
    # std mse (rsfmri)
    std_mse_rsfmri_train = np.array([])
    std_mse_rsfmri_test = np.array([])
    # std rmse (rsfmri)
    std_rmse_rsfmri_train = np.array([])
    std_rmse_rsfmri_test = np.array([])

    for dim in dimensions:
        # Save MSE, RMSE (gyr + rsfmr)
        cvscores_mse_train = [
            np.load("{}/{}/fold_{}/cvscores_mse_train.npy".format(base_path, dim, fold))
            for fold in range(1, nb_folds + 1)
        ]

        cvscores_mse_test = [
            np.load("{}/{}/fold_{}/cvscores_mse_test.npy".format(base_path, dim, fold))
            for fold in range(1, nb_folds + 1)
        ]

        cvscores_rmse_train = [
            np.load(
                "{}/{}/fold_{}/cvscores_rmse_train.npy".format(base_path, dim, fold)
            )
            for fold in range(1, nb_folds + 1)
        ]

        cvscores_rmse_test = [
            np.load("{}/{}/fold_{}/cvscores_rmse_test.npy".format(base_path, dim, fold))
            for fold in range(1, nb_folds + 1)
        ]
        np.append(mse_train, np.mean(cvscores_mse_train))
        np.append(std_mse_train, np.std(cvscores_mse_train))
        np.append(mse_test, np.mean(cvscores_mse_test))
        np.append(std_mse_test, np.std(cvscores_mse_test))
        np.append(rmse_train, np.mean(cvscores_rmse_train))
        np.append(std_rmse_train, np.std(cvscores_rmse_train))
        np.append(rmse_test, np.mean(cvscores_rmse_test))
        np.append(std_rmse_test, np.std(cvscores_rmse_test))

        # Save MSE, RMSE (gyr)
        cvscores_mse_gyr_train = [
            np.load(
                "{}/{}/fold_{}/cvscores_mse_gyr_train.npy".format(base_path, dim, fold)
            )
            for fold in range(1, nb_folds + 1)
        ]

        cvscores_mse_gyr_test = [
            np.load(
                "{}/{}/fold_{}/cvscores_mse_gyr_test.npy".format(base_path, dim, fold)
            )
            for fold in range(1, nb_folds + 1)
        ]
        cvscores_rmse_gyr_train = [
            np.load(
                "{}/{}/fold_{}/cvscores_rmse_gyr_train.npy".format(base_path, dim, fold)
            )
            for fold in range(1, nb_folds + 1)
        ]
        cvscores_rmse_gyr_test = [
            np.load(
                "{}/{}/fold_{}/cvscores_rmse_gyr_test.npy".format(base_path, dim, fold)
            )
            for fold in range(1, nb_folds + 1)
        ]
        np.append(mse_gyr_train, np.mean(cvscores_mse_gyr_train))
        np.append(std_mse_gyr_train, np.std(cvscores_mse_gyr_train))
        np.append(mse_gyr_test, np.mean(cvscores_mse_gyr_test))
        np.append(std_mse_gyr_test, np.std(cvscores_mse_gyr_test))
        np.append(rmse_gyr_train, np.mean(cvscores_rmse_gyr_train))
        np.append(std_rmse_gyr_train, np.std(cvscores_rmse_gyr_train))
        np.append(rmse_gyr_test, np.mean(cvscores_rmse_gyr_test))
        np.append(std_rmse_gyr_test, np.std(cvscores_rmse_gyr_test))

        # Save MSE, RMSE (rsfmri)
        cvscores_mse_rsfmri_train = [
            np.load(
                "{}/{}/fold_{}/cvscores_mse_rsfmri_train.npy".format(
                    base_path, dim, fold
                )
            )
            for fold in range(1, nb_folds + 1)
        ]

        cvscores_mse_rsfmri_test = [
            np.load(
                "{}/{}/fold_{}/cvscores_mse_rsfmri_test.npy".format(
                    base_path, dim, fold
                )
            )
            for fold in range(1, nb_folds + 1)
        ]

        cvscores_rmse_rsfmri_train = [
            np.load(
                "{}/{}/fold_{}/cvscores_rmse_rsfmri_train.npy".format(
                    base_path, dim, fold
                )
            )
            for fold in range(1, nb_folds + 1)
        ]

        cvscores_rmse_rsfmri_test = [
            np.load(
                "{}/{}/fold_{}/cvscores_rmse_rsfmri_test.npy".format(
                    base_path, dim, fold
                )
            )
            for fold in range(1, nb_folds + 1)
        ]
        np.append(mse_rsfmri_train, np.mean(cvscores_mse_rsfmri_train))
        np.append(std_mse_rsfmri_train, np.std(cvscores_mse_rsfmri_train))
        np.append(mse_rsfmri_test, np.mean(cvscores_mse_rsfmri_test))
        np.append(std_mse_rsfmri_test, np.std(cvscores_mse_rsfmri_test))
        np.append(rmse_rsfmri_train, np.mean(cvscores_rmse_rsfmri_train))
        np.append(std_rmse_rsfmri_train, np.std(cvscores_rmse_rsfmri_train))
        np.append(rmse_rsfmri_test, np.mean(cvscores_rmse_rsfmri_test))
        np.append(std_rmse_rsfmri_test, np.std(cvscores_rmse_rsfmri_test))

    # load MSE, RMSE, and STD vectors for training and test sets
    np.save("{}/mse_train_mean.npy".format(base_path), np.array([mse_train]))
    np.save("{}/rmse_train_mean.npy".format(base_path), np.array([rmse_train]))
    np.save("{}/std_mse_train_mean.npy".format(base_path), np.array([std_mse_train]))
    np.save("{}/std_rmse_train_mean.npy".format(base_path), np.array([std_rmse_train]))
    np.save("{}/mse_test_mean.npy".format(base_path), np.array([mse_test]))
    np.save("{}/rmse_test_mean.npy".format(base_path), np.array([rmse_test]))
    np.save("{}/std_mse_test_mean.npy".format(base_path), np.array([std_mse_test]))
    np.save("{}/std_rmse_test_mean.npy".format(base_path), np.array([std_rmse_test]))

    # load MSE, RMSE, and STD vectors for training and test sets (rsfmri)

    np.save(
        "{}/mse_test_mean_rsfmri.npy".format(base_path), np.array([mse_rsfmri_test]),
    )
    np.save(
        "{}/rmse_test_mean_rsfmri.npy".format(base_path), np.array([rmse_rsfmri_test]),
    )
    np.save(
        "{}/mse_train_mean_rsfmri.npy".format(base_path), np.array([mse_rsfmri_train]),
    )
    np.save(
        "{}/rmse_train_mean_rsfmri.npy".format(base_path),
        np.array([rmse_rsfmri_train]),
    )
    np.save(
        "{}/std_mse_mean_rsfmri.npy".format(base_path), np.array([std_mse_rsfmri_test]),
    )
    np.save(
        "{}/std_rmse_mean_rsfmri.npy".format(base_path),
        np.array([std_rmse_rsfmri_test]),
    )

    # plotting the mse train
    # setting x and y axis range
    # plotting the mse train

    # Here what we really want is all the combinations between dimensions_1 and 2, and not just dimensions_1
    plt.plot(dimensions, mse_train, label="mse_train")
    plt.plot(dimensions, mse_test, label="mse_test")
    plt.xlabel("Encoding dimension")
    plt.ylabel("Reconstruction error (MSE)")
    # showing legend
    plt.legend()
    plt.savefig("{}/reconstruction_error_mse.pdf".format(base_path))
    plt.savefig("{}/reconstruction_error_mse.png".format(base_path))
    plt.close()
    # plotting the rmse train
    # setting x and y axis range
    plt.plot(dimensions, rmse_train, label="rmse_train")
    plt.plot(dimensions, rmse_test, label="rmse_test")
    plt.xlabel("Encoding dimension")
    plt.ylabel("Reconstruction error (RMSE)")
    # showing legend
    plt.legend()
    plt.savefig("{}/reconstruction_error_rmse.pdf".format(base_path))
    plt.savefig("{}/reconstruction_error_rmse.png".format(base_path))
    plt.close()


if __name__ == "__main__":
    # The dimensions are used accross 3 scripts, there should be a parameter file that is loaded, probably in json format
    # dimensions = ["15-5"]
    dimensions_1 = [18, 17, 16, 15, 14, 13, 12, 11, 10]
    dimensions_2 = [2, 3, 4, 6, 5, 7, 8, 9, 10]
    nb_folds = 10
    dimensions = np.array([])
    for dim_1 in dimensions_1:
        for dim_2 in dimensions_2:
            dimensions = np.append(dimensions, str(dim_1) + "-" + str(dim_2))
    # data_orig = sys.argv[1]
    # data_type = sys.argv[2]  # could be "tfMRI" or "gyrification"
    paths_to_analyse = [
        "/scratch/mmahaut/data/intertva/ae",
        # "/scratch/mmahaut/data/intertva/ae_gyrification",
        # "/scratch/mmahaut/data/abide/ae_gyrification",
    ]
    titles = [
        "interTVA tfMRI MSE",
        # "interTVA gyrification MSE",
        # "ABIDE gyrification MSE",
    ]
    stat_files = [
        "rmse_test_mean.npy",
        "rmse_test_mean_rsfmri.npy",
        "rmse_train_mean.npy",
        "rmse_train_mean_rsfmri.npy",
        "std_mse_mean_rsfmri.npy",
        "std_mse_test_mean.npy",
        "std_mse_train_mean.npy",
        "std_rmse_mean_rsfmri.npy",
        "std_rmse_test_mean.npy",
        "std_rmse_train_mean.npy",
    ]
    for i in range(len(paths_to_analyse)):
        regroup_stats(paths_to_analyse[i], dimensions, nb_folds)
        table = build_stat_table(dimensions, paths_to_analyse[i], stat_files, titles[i])

