# This scripts will give the mse and r squared in two columns of different modalities chosen.
# If different hyperparameters are used, and need to be taken into account during path building,
# regression_stats_hyperparameters should be used.


import numpy as np
import sys
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # The dimensions are used accross 3 scripts, there should be a parameter file that is loaded, probably in json format
    dimensions = ["15-5"]
    fold_number = 8
    path = "/scratch/mmahaut/data/intertva/regression_output"

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    cell_text = []
    rows = [
        # "tfMRI + rsfMRI",
        # "Anat + rsfMRI",
        "raw_tfMRI + raw_rsfMRI",
        "raw_Anat + raw_rsfMRI",
    ]
    columns = ["Average MSE", "Average R²"]
    title = "InterTVA trace-regression quantitative evaluation from relu/linear 20 dimensional multimodal auto-encoder"
    for dim in dimensions:
        for modality in [
            # "tfMRI",
            # "gyrification",
            "raw_input/tfMRI",
            "raw_input/gyrification",
        ]:
            mse = []
            r_squared = []
            for fold in range(1, fold_number + 1):
                full_path = os.path.join(
                    path, modality, str(dim), "fold_{}".format(fold)
                )
                print(full_path)
                mse.append(np.load(os.path.join(full_path, "mse.npy")))
                r_squared.append(np.load(os.path.join(full_path, "r_squared.npy")))
            print("MSE : ", mse)
            print("MSE mean", np.mean(mse))
            # print("[MSE] mean", np.mean([mse])) # <-- same thing as above, just needed to be sure
            cell_text.append(
                [
                    "%.3f (+/- %.5f)" % (np.mean(mse), np.std(mse)),
                    "%.3f (+/- %.5f)" % (np.mean(r_squared), np.std(r_squared)),
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

    # mse = []
    # rsquared = []
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
