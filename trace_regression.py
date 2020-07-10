import numpy as np
import convexminimization as cvm
import pickle
import scipy.sparse as ssp
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# import matplotlib.pyplot as plt


def load_data(fold):
    # Les données dans une liste (pour commencer)
    X = []
    for i in range(3, 43):
        if i == 36:  # Avoid missing data
            continue
        mat = np.load("fold_{}/X_{}.npy".format(fold, i))
        X.append(mat)

    XZ = np.array(X)

    Y = [81.25, 81.25, 93.75, 93.75, 93.75, 62.5, 81.25, 100, 100, 87.5, 87.5, 68.75, 68.75, 87.5, 93.75, 100, 62.5,
         87.5, 93.75, 87.5, 81.25, 81.25, 81.25, 93.75, 50, 62.5, 93.75, 81.25, 81.25, 87.5, 68.75, 81.25, 87.5, 87.5,
         87.5, 75, 93.75, 93.75, 93.75]
    YZ = np.array(Y)
    x = np.array(Y)
    YZ = (x - min(x)) / (max(x) - min(x))
    return XZ, YZ


def load_graph():
    graph = None
    with open("adj_matrix.pck", "rb") as f:
        graph = pickle.load(f)
    return graph


def create_objective_function(X, Y, graph, delta):
    def objective_function(beta):
        val = 0.0
        for i in range(len(X)):
            val += 0.5 * (np.trace(X[i].T @ beta) - Y[i]) ** 2.0
        val += delta * 0.5 * np.trace(beta.T @ graph @ beta)
        return val

    return objective_function


def create_objective_function_vec(X, Y, graph, delta):
    def objective_function(beta):
        val = 0.5 * ((np.trace(np.transpose(X, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2) - Y) ** 2).sum()
        val += delta * 0.5 * np.trace(beta.T @ graph @ beta)
        return val

    return objective_function


def create_gradient_function(X, Y, graph, delta):
    def gradient_function(beta):
        grad = beta * 0.0
        for i in range(len(X)):
            grad += X[i] * (np.trace(X[i].T @ beta) - Y[i])
        grad += delta * graph @ beta
        return grad

    return gradient_function


def create_gradient_function_vec(X, Y, graph, delta):
    def gradient_function(beta):
        grad = (X * (np.trace(np.transpose(X, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2) - Y)[:, np.newaxis,
                    np.newaxis]).sum(axis=0)
        grad += delta * graph @ beta
        return grad

    return gradient_function


# The identity projector, i.e no constraints
def identity_projector(beta, mu):
    return beta.copy()


# The ridge projector, i.e l2 ball constraints
def create_ridge_projector(rho):
    def ridge_projector(beta, mu):
        norm_beta = np.linalg.norm(beta)
        if norm_beta <= rho:
            return beta.copy()
        return beta / norm_beta * rho


# Group sparsity projector, here sparsity on the lines
def create_group_sparsity_projector(delta):
    def group_sparsity_projector(beta, mu):
        norms = np.linalg.norm(beta, axis=1)
        idx = np.where(norms > delta)
        res = beta * 0.0
        res[idx, :] = beta[idx, :] - np.squeeze(np.sign(beta[idx, :])) * delta * mu / norms[idx, np.newaxis]
        return res

    return group_sparsity_projector


# Estimate beta from data
def estimate_beta(X, Y, params):
    objective = create_objective_function_vec(X, Y, params["graph"], params["delta"])
    gradient = create_gradient_function_vec(X, Y, params["graph"], params["delta"])
    sparse_projector = create_group_sparsity_projector(params["soft_thresh"])

    (res, mu) = cvm.monotone_fista_support(objective, gradient, X[4] * 0.0, params["mu"],
                                           params["mu_min"], params["iterations"], sparse_projector)
    return res


def cross_validation_error(X, Y, params, nbfolds=5):
    idx = np.arange(Y.shape[0])
    np.random.shuffle(idx)  # Met le bazar dans les indices
    spls = np.array_split(idx, nbfolds)  # Découpe en plusieurs morceaux
    results = np.zeros(Y.shape[0])
    for spl in spls:
        reste = np.setdiff1d(np.arange(Y.shape[0]), spl)

        # Ensemble d'entrainement
        XE = X[reste, :, :]
        YE = Y[reste]

        # Ensemble de test
        XT = X[spl, :, :]
        YT = Y[spl]

        beta = estimate_beta(XE, YE, params)

        # Estimate the results
        results[spl] = np.trace(np.transpose(XT, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2)

    return results

mse=[]
rsquared=[]
# Calcul du Laplacien du graphe
graph = load_graph()
degree = np.array(graph.sum(axis=0))
laplacian = ssp.diags(degree, offsets=[0]) - graph

# Les paramètres
params = {}
params["iterations"] = 1000
params["mu"] = 0.1
params["mu_min"] = 1e-7
# params["soft_thresh"] = 10e-3
params["soft_thresh"] = 0.0
params["delta"] = 0.0
params["graph"] = laplacian
if __name__ == "__main__":
    # 10-fold validation
    idx = np.arange(39)
    kf = KFold(n_splits=10)
    fold = 0
    results = np.zeros(39)
    for train_index, test_index in kf.split(idx):
        fold += 1
        print(f"Fold #{fold}")
        # Chargement des données
        X, Y = load_data(fold)
        print("TRAIN:", idx[train_index], "TEST:", idx[test_index])
        # Ensemble d'entrainement
        XE = X[idx[train_index], :, :]
        YE = Y[idx[train_index]]
        # Ensemble de test
        XT = X[idx[test_index],:,:]
        YT = Y[idx[test_index]]
        beta = estimate_beta(XE, YE, params)
        file = "fold_{}/beta.npy".format(fold)
        np.save(file, beta)
        # Estimate the results
        results[idx[test_index]] = np.trace(np.transpose(XT, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2)
        print(results[idx[test_index]])
        print("MSE, fold_{}".format(fold), mean_squared_error(YT, results[idx[test_index]]))
        print("R2 score, fold_{}".format(fold), r2_score(YT, results[idx[test_index]]))
        file = "fold_{}/mse.npy".format(fold)
        np.save(file, mean_squared_error(YT, results[idx[test_index]]))
        file = "fold_{}/r_squared.npy".format(fold)
        np.save(file, r2_score(YT, results[idx[test_index]]))
        mse.append([mean_squared_error(YT, results[idx[test_index]])])
        rsquared.append([r2_score(YT, results[idx[test_index]])])
 

print("mean mse {}".format(np.mean([mse])))
file = "mean_mse.npy"
np.save(file, np.mean([mse]))
print("mean r squared {}".format(np.mean([rsquared])))
file = "mean_rsquared.npy"
np.save(file, np.mean([rsquared]))
print(results)
print("Mean Error = {}".format(np.linalg.norm(results - Y) ** 0.2 / Y.shape[0]))
print("MSE = {}".format( mean_squared_error(Y, results)))
file = "mse.npy"
np.save(file,mean_squared_error(Y, results))
file = "r2_score.npy"
np.save(file,r2_score(Y, results))














