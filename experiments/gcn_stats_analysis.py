import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/scratch/mmahaut/scripts/INT_fMRI_processing/")

from regression import build_raw_xy_data, load_graph
from mdae_step import build_path_and_vars

if __name__ == "__main__":

    y_prediction = np.zeros(39)
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
    x = np.array(Y)
    y_val = (x - min(x)) / (max(x) - min(x))
    params = {}

    params["data_source"] = "interTVA"  # ABIDE or interTVA
    params["modality"] = "tfMRI"  # gyrification or tfMRI
    # According to IJCNN paper, 15 is the best number of dimensions for tfMRI
    params["dim_1"] = 15
    # According to IJCNN paper, 5 is the best number of dimensions for rsfMRI
    params["dim_2"] = 5

    sub_index = np.arange(0, len(sub_list))

    for fold in range(1, 11):
        (
            train_index,
            test_index,  # <- train and test index are loaded to match mdae training
            params["ref_subject"],
            params["orig_path"],
            params["base_path"],
            sub_list,
        ) = build_path_and_vars(
            params["data_source"],
            params["modality"],
            str(params["dim_1"]) + "-" + str(params["dim_2"]),
            fold,
        )
        y_prediction[sub_index[test_index]] = np.load(
            "/scratch/mmahaut/data/intertva/ae/gcnn/fold_{}/predictions.npy".format(
                fold
            )
        )

    regr = linear_model.LinearRegression()
    regr.fit(y_val.reshape(-1, 1), y_prediction)
    y_reg = regr.predict(y_val.reshape(-1, 1))
    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_val, y_reg))
    # The coefficient of determination: 1 is perfect prediction
    print("R²: %.2f" % r2_score(y_val, y_reg))
    file_path = "/scratch/mmahaut/data/intertva/ae/gcnn/"

    plt.scatter(y_val, y_prediction, color="black")
    droite = plt.plot(
        y_val,
        y_reg,
        color="blue",
        label="Coefficients: {}\nMean squared error: %.2f \nR²: %.2f".format(regr.coef_)
        % (mean_squared_error(y_val, y_reg), r2_score(y_val, y_reg)),
    )
    plt.xlabel("Y réel")
    plt.ylabel("Y prédit")
    plt.title("GCN direct regression\ninterTVA tfMRI+rsfMRI")
    plt.legend()

    plt.savefig(os.path.join(file_path, "y_ypred.png"))
