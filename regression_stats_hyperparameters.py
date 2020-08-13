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
    dimensions = ["15-5"]
    fold_number = 3
    path = "/scratch/mmahaut/data/intertva/regression_output"

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    mse = []
    rsquared = []

    # # STATS (for later) (taken from the end of regression, apart from folds)
    # print("mean mse {}".format(np.mean([mse])))
    # file_path = "{}/regression_output/".format(params["orig_path"])
    # file = "mean_mse.npy"
    # np.save(os.path.join(file, file_path), np.mean([mse]))
    # print("mean r squared {}".format(np.mean([rsquared])))
    # file = "mean_rsquared.npy"
    # np.save(os.path.join(file, file_path), np.mean([rsquared]))
    # print(results)
    # print("Mean Error = {}".format(np.linalg.norm(results - Y) ** 0.2 / Y.shape[0]))
    # print("MSE = {}".format(mean_squared_error(Y, results)))
    # file = "mse.npy"
    # np.save(os.path.join(file, file_path), mean_squared_error(Y, results))
    # file = "r2_score.npy"
    # np.save(os.path.join(file, file_path), r2_score(Y, results))
    titles = [
        "tfMRI",
        "gyrification",
        "raw_input_tfMRI",
        "raw_input_gyrification",
    ]
    i = 0
    # Here rows are for soft thres params
    rows = [1e-5, 1e-6, 1e-7, 1e-8]
    # Here collumns are for delta params
    columns = [1e-3, 1e-6, 1e-7, 1e-8]
    for dim in dimensions:
        for modality in [
            "tfMRI",
            "gyrification",
            "raw_input/tfMRI",
            "raw_input/gyrification",
        ]:

            cell_text = []
            for soft_thres in rows:
                cell_text_row = []
                for delta in columns:
                    mse = []
                    r_squared = []
                    for fold in range(1, fold_number + 1):
                        full_path = os.path.join(
                            path,
                            modality,
                            str(dim),
                            "fold_{}".format(fold),
                            "delta_{}".format(delta),
                            "soft_thres_{}".format(soft_thres),
                        )
                        # print(full_path)
                        mse.append(np.load(os.path.join(full_path, "mse.npy")))
                        r_squared.append(
                            np.load(os.path.join(full_path, "r_squared.npy"))
                        )
                    print("MSE : ", mse)
                    print("MSE mean", np.mean(mse))

                    cell_text_row.append(
                        str(
                            "%.3f (+/- %.5f)" % (np.mean(mse), np.std(mse))
                            + "// %.3f (+/- %.5f)"
                            % (np.mean(r_squared), np.std(r_squared))
                        )
                    )

                cell_text.append(cell_text_row)

            title = "{} trace-regression hyperparameter evaluation".format(modality)
            the_table = plt.table(
                cellText=cell_text, rowLabels=rows, colLabels=columns, loc="center",
            )
            the_table.scale(4, 2.5)
            plt.draw()
            plt.title(title)

            plt.savefig(
                os.path.join(
                    path, "regression_hyperparameters_{}.png".format(titles[i])
                ),
                dpi=fig.dpi,
                bbox_inches="tight",
                pad_inches=0.5,
            )
            i += 1

