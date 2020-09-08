import glob
from spektral.layers import GraphConv
from tensorflow.keras import Model
from regression import build_raw_xy_data, load_graph
from mdae_step import build_path_and_vars
import numpy as np
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

y_prediction = np.zeros(39)

for fold in range(1, 11):
    # Calcul du Laplacien du graphe
    A = load_graph()
    params = {}

    params["data_source"] = "interTVA"  # ABIDE or interTVA
    params["modality"] = "tfMRI"  # gyrification or tfMRI
    # According to IJCNN paper, 15 is the best number of dimensions for tfMRI
    params["dim_1"] = 15
    # According to IJCNN paper, 5 is the best number of dimensions for rsfMRI
    params["dim_2"] = 5

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
    sub_index = np.arange(0, len(sub_list))

    # Load data
    X_val, y_val = build_raw_xy_data(params, fold, sub_list)
    X_test = X_val[sub_index[test_index], :, :]
    y_test = y_val[sub_index[test_index]]
    model = load_model(
        ("/scratch/mmahaut/data/intertva/ae/gcnn/fold_{}/model.h5".format(fold)),
        custom_objects={"GraphConv": GraphConv},
    )
    y_prediction[sub_index[test_index]] = model.predict(X_test)

regr = linear_model.LinearRegression()
regr.fit(y_val.reshape(-1, 1), y_prediction)
y_reg = regr.predict(y_val.reshape(-1, 1))
# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_val, y_reg))
# The coefficient of determination: 1 is perfect prediction
print("R²: %.2f" % r2_score(y_val, y_reg))
file_path = os.path.join(
    params["orig_path"], "regression_output/tfMRI/reproducibility/tfmri10-rsfmri10/",
)

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

