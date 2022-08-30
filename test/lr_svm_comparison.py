import os, sys

sys.path.insert(0, os.path.abspath(""))

from source.floating_logistic_regression.flp_logistic import LogisticRegression
from source.floating_svm.flp_dual_svm_ls import FlpDualLSSVM

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(filename):
    df = pd.read_csv("" + filename)
    X = df.iloc[:, :df.shape[1] - 1]
    y = df.iloc[:, df.shape[1] - 1]
    y = np.expand_dims(y, axis=1)
    return X.values, y


if __name__ == "__main__":
    # Read the dataset
    X, y = load_dataset("source/experiments/real_experiment/datasets/credit_curated.csv")
    print("==> Original dataset shapes:")
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("==> Splited dataset shapes:")
    print("Shape of X_train", X_train.shape)
    print("Shape of X_test", X_test.shape)
    print("Shape of y_train", y_train.shape)
    print("Shape of y_test", y_test.shape)
    
    # Changin response variables for logistic regression
    # from -1/1 to 0/1
    y_train_lr = y_train.copy()
    y_train_lr[y_train_lr == -1] = 0
    y_test_lr = y_test.copy()
    y_test_lr[y_test_lr == -1] = 0
    
    # Training LogReg
    log_model = LogisticRegression(lr=0.001, max_iter=2000)
    log_model.fit(X_train, y_train)
    print("==> Logistic regression results:")
    print("Training acc:", log_model.score(X_train, y_train_lr))
    print("Testing acc:", log_model.score(X_test, y_test_lr))
    
    # Training SVM
    svm_model = FlpDualLSSVM(lambd=0.1, lr=0.001, max_iter=2000)
    svm_model.fit(X_train, y_train)
    print("==> SVM results:")
    print("Training acc:", svm_model.score(X_train, y_train))
    print("Testing acc:", svm_model.score(X_test, y_test))