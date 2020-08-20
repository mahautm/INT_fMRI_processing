# This is a script that saves a table for the different hyperparameters tested
# When changing parameters, Iis important to check path, rows, columns (which are bothe used to make paths)
# modality, number of folds,and possibly titles.

import numpy as np
import sys
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # The dimensions are used accross 3 scripts, there should be a parameter file that is loaded, probably in json format
    dimensions = ["15-5"]
    fold_number = 3  # can be lower thant the actual number of folds calculated during regression, but not higher
    path = "/scratch/mmahaut/data/intertva/regression_output"
    titles = [
        "tfMRI",
        "gyrification",
        "raw_input_tfMRI",
        "raw_input_gyrification",
    ]  # Exists solely because using "/"" in the title screws up saving and so I could not use the "modality" variable

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    mse = []
    rsquared = []

    i = 0
    # Here collumns are for delta params
    columns = [1e-3, 1e-2, 1e-1, 1]
    # Here rows are for soft thres params
    rows = [5e-4, 1e-4, 1e-3, 1e-2, 1e-1, 1]

    for dim in dimensions:
        for modality in [
            "tfMRI",
            "gyrification",
            "raw_input/tfMRI",
            "raw_input/gyrification",
        ]:  # USED as an addition to the path variable

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
                        if os.path.exists(full_path):

                            mse.append(np.load(os.path.join(full_path, "mse.npy")))
                            r_squared.append(
                                np.load(os.path.join(full_path, "r_squared.npy"))
                            )
                        else:
                            print("missing : ", full_path)
                    print("MSE : ", mse)
                    print("MSE mean", np.mean(mse))

                    cell_text_row.append(
                        str(
                            "%.3f (+/- %.5f)" % (np.mean(mse), np.std(mse))
                            + "// %.3f (+/- %.5f)"
                            % (
                                np.mean(r_squared),
                                np.std(r_squared),
                            )  # Really not beautiful, and lacks legend...
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

