# This script has to be run AFTER mdae_stats.py, it will summarise the evaluation data in a table and save it as a png
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
            stat = np.load(os.path.join(orig_path, filename))[0]
            row_cell_text.append(stat[dim_index])
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


if __name__ == "__main__":
    # The dimensions are used accross 3 scripts, there should be a parameter file that is loaded, probably in json format
    # dimensions = ["15-5"]
    dimensions_1 = [18, 17, 16, 14, 13, 12, 11, 10]
    dimensions_2 = [2, 3, 4, 6, 7, 8, 9, 10]
    dimensions = np.array([])

    # data_orig = sys.argv[1]
    # data_type = sys.argv[2]  # could be "tfMRI" or "gyrification"
    paths_to_analyse = [
        "/scratch/mmahaut/data/intertva/ae",
        "/scratch/mmahaut/data/intertva/ae_gyrification",
        "/scratch/mmahaut/data/abide/ae_gyrification",
    ]
    titles = [
        "interTVA tfMRI MSE",
        "interTVA gyrification MSE",
        "ABIDE gyrification MSE",
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
        table = build_stat_table(dimensions, paths_to_analyse[i], stat_files, titles[i])

