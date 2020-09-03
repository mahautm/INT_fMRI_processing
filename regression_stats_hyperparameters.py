# This is a script that saves a table for the different hyperparameters tested
# When changing parameters, Iis important to check path, rows, columns (which are bothe used to make paths)
# modality, number of folds,and possibly titles.

import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from mdae_step import build_path_and_vars

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

if __name__ == "__main__":
    # The dimensions are used accross 3 scripts, there should be a parameter file that is loaded, probably in json format
    dimensions = ["15-5"]

    fold_number = 10  # can be lower thant the actual number of folds calculated during regression, but not higher
    path = "/scratch/mmahaut/data/intertva/regression_output"
    titles = [
        "tfMRI",
        # "gyrification",
        # "raw_input_tfMRI",
        # "raw_input_gyrification",
    ]  # Exists solely because using "/"" in the title screws up saving and so I could not use the "modality" variable
    Y = [
        81.25,
        81.25,
        93.75,
        93.75,
        93.75,
        62.5,
        81.25,
        100,
        100,
        87.5,
        87.5,
        68.75,
        68.75,
        87.5,
        93.75,
        100,
        62.5,
        87.5,
        93.75,
        87.5,
        81.25,
        81.25,
        81.25,
        93.75,
        50,
        62.5,
        93.75,
        81.25,
        81.25,
        87.5,
        68.75,
        81.25,
        87.5,
        87.5,
        87.5,
        75,
        93.75,
        93.75,
        93.75,
    ]
    y = np.array(Y)
    Y = (y - min(y)) / (max(y) - min(y))
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    mse = []
    rsquared = []

    i = 0
    # Here collumns are for delta params
    columns = [0.1, 0, 1, 100, 1000]
    # Here rows are for soft thres params
    rows = [0, 1.9e-4, 1.95e-4, 2e-4]

    for dim in dimensions:
        for modality in [
            # "tfMRI",
            "tfMRI/reproducibility",
            # "gyrification",
            # "raw_input/tfMRI",
            # "raw_input/gyrification",
        ]:  # USED as an addition to the path variable

            cell_text = []
            for soft_thres in rows:
                cell_text_row = []
                for delta in columns:
                    mse = []
                    r_squared = []
                    results = np.zeros(39)
                    for fold in range(1, fold_number + 1):
                        full_path = os.path.join(
                            path,
                            modality,
                            str(dim),
                            "fold_{}".format(fold),
                            "delta_{}".format(float(delta)),
                            "soft_thres_{}".format(float(soft_thres)),
                        )
                        if os.path.exists(full_path):
                            (
                                train_index,
                                test_index,  # <- train and test index are loaded to match mdae training
                                ref_subject,
                                orig_path,
                                base_path,
                                sub_list,
                            ) = build_path_and_vars(
                                "interTVA", "tfMRI", str(15) + "-" + str(5), fold,
                            )
                            sub_index = np.arange(0, len(sub_list))
                            results[sub_index[test_index]] = np.load(
                                os.path.join(full_path, "results.npy")
                            )
                            # mse.append(np.load(os.path.join(full_path, "mse.npy")))
                            # r_squared.append(
                            #     np.load(os.path.join(full_path, "r_squared.npy"))
                            # )
                        else:
                            print("missing : ", full_path)
                    # print("MSE : ", mse)
                    # print("MSE mean", np.mean(mse))

                    cell_text_row.append(
                        "%.3f" % mean_squared_error(Y, results)
                        + " // "
                        + "%.3f" % r2_score(Y, results)
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

