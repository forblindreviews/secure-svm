from tkinter.tix import X_REGION
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sympy import N
from torch import threshold

from flp_logistic import LogisticRegression

import os
import sys
sys.path.insert(0, os.path.abspath(""))
from source.experiments.dataset_generator import save_dataset_parties
from source.experiments.dataset_generator import generate_dataset, split_dataset

def load_real_dataset():
    # Load dataset train
    df_train = pd.read_csv("source/experiments/real_experiment/datasets/dataset_train_0.csv")
    X_train = df_train.iloc[:, :df_train.shape[1] - 1].to_numpy()
    y_train = df_train.iloc[:, df_train.shape[1] - 1].to_numpy()

    # Load dataset test
    df_test = pd.read_csv("source/experiments/real_experiment/datasets/dataset_test_0.csv")
    X_test = df_test.iloc[:, :df_test.shape[1] - 1].to_numpy()
    y_test = df_test.iloc[:, df_test.shape[1] - 1].to_numpy()

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    save_dataset_parties(X_train, y_train, 4)
    return X_train, y_train, X_test, y_test


def generate_artificial_dataset(n_samples, n_features, train_percentage):
    X, y = generate_dataset(n_samples, n_features, 1.0)
    y[y == -1] = 0
    return split_dataset(X, y, train_percentage)

    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_artificial_dataset(
        n_samples=500, n_features=2, train_percentage=0.70
    )
    logit_model_flp = LogisticRegression(max_iter=5000)
    print("Max in X:", X_train.max())
    logit_model_flp.fit(X_train, y_train)
    
    print("Score train flp:", logit_model_flp.score(X_train, y_train))
    print("Test train flp:", logit_model_flp.score(X_test, y_test))