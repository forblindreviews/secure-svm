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


def grid_search_log_reg(
    X_train,
    X_test,
    y_train,
    y_test, 
    lr_samples,
    threshold_samples
):
    max_test_acc = 0
    max_train_acc = 0 
    max_lr = 0
    max_threshold = 0
    for lr in lr_samples:
        for threshold in threshold_samples:
            model = LogisticRegression(lr, threshold, max_iter=2000)
            model.fit(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            train_acc = model.score(X_train, y_train)
            if max_test_acc < test_acc:
                max_test_acc = test_acc
                max_lr = lr
                max_threshold = threshold
                max_train_acc = train_acc
    return max_train_acc, max_test_acc, max_lr, max_threshold


def grid_search_svm(
    X_train,
    X_test,
    y_train,
    y_test, 
    lambd_samples,
    lr_samples
):
    max_test_acc = 0
    max_train_acc = 0 
    max_lambd = 0
    max_lr = 0
    for lambd in lambd_samples:
        for lr in lr_samples:
            model = FlpDualLSSVM(
                lambd=lambd,
                lr=lr,
                max_iter=500
            )
            model.fit(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            train_acc = model.score(X_train, y_train)
            if max_test_acc < test_acc:
                max_test_acc = test_acc
                max_train_acc = train_acc
                max_lr = lr
                max_lambd = lambd
    return max_train_acc, max_test_acc, max_lr, max_lambd


if __name__ == "__main__":
    # Grid sample LogReg
    logreg_lr_samples = [0.1, 0.05, 0.01, 0.005, 0.001]
    logreg_threshold_samples = [0.5]
    
    # Grid sample SVM
    svm_lambd_samples = [0.001, 0.01, 0.1, 1, 10, 20, 50, 100]
    svm_lr_samples = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    
    # ==> Tunning HCC
    X_hcc_train, y_hcc_train = load_dataset("source/experiments/real_experiment/datasets/dataset_train_0.csv")
    X_hcc_test, y_hcc_test = load_dataset("source/experiments/real_experiment/datasets/dataset_test_0.csv")
    
    # => Tunning LogReg
    y_hcc_train_logreg = y_hcc_train.copy()
    y_hcc_train_logreg[y_hcc_train_logreg == -1] = 0
    y_hcc_test_logreg = y_hcc_test.copy()
    y_hcc_test_logreg[y_hcc_test_logreg == -1] = 0
    
    max_train_acc_logreg, max_test_acc_logreg, max_lr_logreg, max_threshold_logreg = grid_search_log_reg(
        X_hcc_train,
        X_hcc_test,
        y_hcc_train_logreg,
        y_hcc_test_logreg,
        logreg_lr_samples,
        logreg_threshold_samples
    )
    
    print("Results LogReg - HCC Dataset")
    print("     max_lr", max_lr_logreg)
    print("     max_threshold", max_threshold_logreg)
    print("     max_train_acc", max_train_acc_logreg)
    print("     max_test_acc", max_test_acc_logreg)
    
    # # => Tunning LS-SVM   
    # max_train_acc_svm, max_test_acc_svm, max_lr_svm, max_lambd_svm = grid_search_svm(
    #     X_hcc_train,
    #     X_hcc_test,
    #     y_hcc_train,
    #     y_hcc_test,
    #     svm_lambd_samples,
    #     svm_lr_samples
    # )
    
    # print("Results SVM - HCC Dataset")
    # print("     max_lr", max_lr_svm)
    # print("     max_lambd", max_lambd_svm)
    # print("     max_train_acc", max_train_acc_svm)
    # print("     max_test_acc", max_test_acc_svm)
    
    # ==> Tunning Credit card
    X_cre, y_cre = load_dataset("source/experiments/real_experiment/datasets/credit_curated.csv")
    X_cre_train, X_cre_test, y_cre_train, y_cre_test = train_test_split(
        X_cre,
        y_cre,
        test_size=0.3
    )

    # Tunning LogReg
    y_cre_train_logreg = y_cre_train.copy()
    y_cre_train_logreg[y_cre_train_logreg == -1] = 0
    y_cre_test_logreg = y_cre_test.copy()
    y_cre_test_logreg[y_cre_test_logreg == -1] = 0
    
    max_train_acc_logreg, max_test_acc_logreg, max_lr_logreg, max_threshold_logreg = grid_search_log_reg(
        X_cre_train,
        X_cre_test,
        y_cre_train_logreg,
        y_cre_test_logreg,
        logreg_lr_samples,
        logreg_threshold_samples
    )
    
    print("Results LogReg - Credit Dataset")
    print("     max_lr", max_lr_logreg)
    print("     max_threshold", max_threshold_logreg)
    print("     max_train_acc", max_train_acc_logreg)
    print("     max_test_acc", max_test_acc_logreg)
    
    # => Tunning SVM
    # max_train_acc_svm, max_test_acc_svm, max_lr_svm, max_lambd_svm = grid_search_svm(
    #     X_cre_train,
    #     X_cre_test,
    #     y_cre_train,
    #     y_cre_test,
    #     svm_lambd_samples,
    #     svm_lr_samples
    # )
    
    # print("Results SVM - Credit Dataset")
    # print("     max_lr", max_lr_svm)
    # print("     max_lambd", max_lambd_svm)
    # print("     max_train_acc", max_train_acc_svm)
    # print("     max_test_acc", max_test_acc_svm)