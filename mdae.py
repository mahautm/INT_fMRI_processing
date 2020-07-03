import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import sys as os
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.optimizers import SGD, Adadelta, Adam
import numpy as np
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def load_data(
    sub, view, ref_sub="USM_0050475", orig_path="/scratch/mmahaut/data/abide/"
):
    # Import anatomical gyrification data
    if view == 1:

        data_path = os.path.join(
            orig_path,
            "features_gyrification/{}_eig_vec_fsaverage5_onref_{}.npy".format(
                sub, ref_sub
            ),
        )
        view_gyr = np.load(data_path)
        return view_gyr

    # Import Resting-State fMRI data
    if view == 2:
        view_rsfmri = np.load(
            os.path.join(
                orig_path,
                "features_rsfMRI/correlation_matrix_fsaverage5_{}.npy".format(sub),
            )
        )
        return view_rsfmri

    # Import concatenated fMRI data
    if view == 3:
        view_rsfmri = np.load(
            os.path.join(
                orig_path, "rsfmri/{}/correlation_matrix_fsaverage5.npy".format(sub)
            )
        )
        view_tfmri = np.load(
            os.path.join(orig_path, "tfmri/{}/gii_matrix_fsaverage5.npy".format(sub))
        )
        fmri_data = np.concatenate([view_tfmri, view_rsfmri], axis=1)
        return fmri_data
