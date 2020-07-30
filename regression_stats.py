import numpy as np
import sys
import matplotlib.pyplot as plt
import os

# from mdae_step import


def build_stat_table(dimensions, orig_path, stat_files, title=""):
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
    dimensions = [20]
    fold_number = 10
    path = "/scratch/mmahaut/data/intertva/"

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    cell_text = []
    rows = ["tfMRI + rsfMRI", "Anat + rsfMRI"]
    columns = ["Average MSE", "Average R²"]
    title = "InterTVA trace-regression quantitative evaluation from relu/linear 20 dimensional multimodal auto-encoder"

    for modality in ["ae", "ae_gyrification"]:
        mse = []
        r_squared = []
        for fold in range(1, fold_number + 1):
            full_path = os.path.join(
                path, modality, "regression_output/fold_{}".format(fold)
            )
            mse.append(np.load(os.path.join(full_path, "mse.npy")))
            r_squared.append(np.load(os.path.join(full_path, "r_squared.npy")))

        cell_text.append(
            [
                "%.3f%% (+/- %.5f%%)" % (np.mean(mse), np.std(mse)),
                "%.3f%% (+/- %.5f%%)" % (np.mean(r_squared), np.std(r_squared)),
            ]
        )
    the_table = plt.table(
        cellText=cell_text, rowLabels=rows, colLabels=columns, loc="center",
    )
    the_table.scale(4, 2.5)
    plt.draw()
    plt.title(title)
    plt.savefig(
        os.path.join(path, "regression_summary.png"),
        dpi=fig.dpi,
        bbox_inches="tight",
        pad_inches=0.5,
    )

