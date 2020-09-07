import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


sys.path.append("../")
from mdae_step import build_path_and_vars

y_true = [
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
y_true = np.array(y_true)
y_prediction = np.zeros(39)
sub_index = np.arange(0, 39)

kf = KFold(n_splits=10)
print(kf.get_n_splits(sub_index))
print("number of splits:", kf)
orig_path = "/scratch/mmahaut/data/intertva"
fold = 0
for train_index, test_index in kf.split(sub_index):
    fold += 1
    file_path = os.path.join(
        orig_path,
        "regression_output/tfMRI/reproducibility/tfmri10-rsfmri10/fold_{}".format(fold),
    )
    beta = np.load(os.path.join(file_path, "beta.npy"))

    X = []
    for i in range(3, 43):
        if i == 36:  # Avoid missing data
            continue
        mat_tf = np.load(
            "/scratch/mmahaut/data/intertva/past_data/representation_learning/relu_linear_three_layers/tfmri/10/fold_{}/X_{}.npy".format(
                fold, i
            )
        )
        mat_rsf = np.load(
            "/scratch/mmahaut/data/intertva/past_data/representation_learning/relu_linear_three_layers/rsfmri/10/fold_{}/X_{}.npy".format(
                fold, i
            )
        )
        X.append(np.concatenate((mat_tf, mat_rsf), axis=1))

    X = np.array(X)
    print("TRAIN:", sub_index[train_index], "TEST:", sub_index[test_index])
    # Ensemble d'entrainement
    XE = X[sub_index[train_index], :, :]
    # Ensemble de test
    XT = X[sub_index[test_index], :, :]

    # Estimate the results
    y_prediction[sub_index[test_index]] = np.trace(
        np.transpose(XT, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2
    )

regr = linear_model.LinearRegression()
regr.fit(y_true.reshape(-1, 1), y_prediction)
y_reg = regr.predict(y_true.reshape(-1, 1))
# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_true, y_reg))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_true, y_reg))
file_path = os.path.join(
    orig_path, "regression_output/tfMRI/reproducibility/tfmri10-rsfmri10/",
)

plt.scatter(y_true, y_prediction, color="black")
droite = plt.plot(y_true, y_reg, color="blue")
plt.legend(
    [droite],
    [
        "Coefficients: {}\nMean squared error: %.2f \nR²: %.2f".format(regr.coef_)
        % (mean_squared_error(y_true, y_reg), r2_score(y_true, y_reg))
    ],
)

plt.savefig(os.path.join(file_path, "y_ypred.png"))

