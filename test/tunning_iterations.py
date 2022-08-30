import os, sys
sys.path.insert(0, os.path.abspath(""))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from source.floating_logistic_regression.flp_logistic import LogisticRegression
from source.floating_svm.flp_dual_svm_ls import FlpDualLSSVM
from source.experiments.dataset_generator import generate_dataset


def load_dataset(filename):
    df = pd.read_csv("" + filename)
    X = df.iloc[:, :df.shape[1] - 1]
    y = df.iloc[:, df.shape[1] - 1]
    y = np.expand_dims(y, axis=1)
    return X.values, y


if __name__ == "__main__":
    # ==> Dataset HCC
    
    X_hcc_train, y_hcc_train = load_dataset("source/experiments/real_experiment/datasets/credit_curated_train_0.csv")
    X_hcc_test, y_hcc_test = load_dataset("source/experiments/real_experiment/datasets/credit_curated_test_0.csv")
    
    # => Tunning LogReg
    y_hcc_train_logreg = y_hcc_train.copy()
    y_hcc_train_logreg[y_hcc_train_logreg == -1] = 0
    y_hcc_test_logreg = y_hcc_test.copy()
    y_hcc_test_logreg[y_hcc_test_logreg == -1] = 0
    
    # model = LogisticRegression(lr=0.1, threshold=0.5, max_iter=2000)
    model = FlpDualLSSVM(lr=0.001, lambd=20, max_iter=500)
    
    model.fit(X_hcc_train, y_hcc_train_logreg)
    
    # errors = np.array(model.info["errors"])
    errors = np.array(model.info["pk_norm"])
    
    fig = plt.figure()
    ax = plt.axes()
    ax.set_yscale("log")
    ax.plot(errors)
    plt.show()
    