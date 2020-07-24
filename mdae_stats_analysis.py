import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


def build_stat_table(dimensions, orig_path, stat_files):
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
            stat = np.load(os.path.join(orig_path, filename))
            row_cell_text.append(stat[dim_index])
            columns.append(
                filename[: len(filename) - 5]
            )  # enlever le .npy à la fin des noms de fichier
        cell_text.append(row_cell_text)
    the_table = plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns,)

    fig.tight_layout()
    return the_table


if __name__ == "__main__":
    # The dimensions are used accross 3 scripts, there should be a parameter file that is loaded, probably in json format
    dimensions = [20]
    # data_orig = sys.argv[1]
    # data_type = sys.argv[2]  # could be "tfMRI" or "gyrification"
    paths_to_analyse = [
        "/scratch/mmahaut/data/intertva/ae",
        "/scratch/mmahaut/data/intertva/ae_gyrification",
        "/scratch/mmahaut/data/abide/ae_gyrification",
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
    for orig_path in paths_to_analyse:
        table = build_stat_table(dimensions, orig_path, stat_files)
        plt.title("mse_summary")
        plt.savefig(os.path.join(orig_path, "mse_summary.png"))
