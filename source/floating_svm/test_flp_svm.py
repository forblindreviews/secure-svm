import sys, os
from sklearn import datasets
from sympy import degree
sys.path.insert(0, os.path.abspath(""))

import source.experiments.dataset_generator as data
import source.analyzers.analyzer_utils as analyzer 

from numpy.lib.npyio import load
from sklearn import datasets
import numpy as np
#import flp_svm
import flp_dual_svm
import flp_dual_svm_ls_scaled
import flp_svm
#import flp_dual_svm_simp
#import flp_dual_svm_fast
#import flp_dual_svm_mix
import flp_dual_svm_ls
import flp_dual_svm_ls_BB88
#import flp_dual_svm_gs
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math

def generate_dataset(n_samples, n_features):
    X, y = datasets.make_classification(n_samples, n_features, n_redundant=0, n_informative=2, class_sep=12)
    y = pd.Series(y).map({0: -1, 1: 1}).values

    # Extend y columns
    y = np.expand_dims(y, axis=1)

    # Save dataset for MATLAB testing
    df_save = pd.DataFrame(data=np.append(X, y, axis=1))
    df_save.to_csv("source/datasets/toy_dataset.csv", index=False, columns=None)

    return X, y

def load_dataset(filename):
    df = pd.read_csv("" + filename)
    X = df.iloc[:, :df.shape[1] - 1]
    y = df.iloc[:, df.shape[1] - 1]

    y = np.expand_dims(y, axis=1)

    return X.values, y

#X, y = generate_dataset(40, 2) 
X, y = load_dataset("source/experiments/real_experiment/datasets/credit_curated_train_0.csv")

# Print shape of dataset
print("X shape =", X.shape)
print("y shape =", y.shape)

# data.save_dataset_parties(X, y, 4)
# data_train = np.concatenate((X, y), axis=1)
# pd.DataFrame(data_train).to_csv("source/analyzers/performance_secure_tester/train.csv", index=False)
# pd.DataFrame(data_train).to_csv("source/analyzers/performance_secure_tester/test.csv", index=False)

# svm_ls = flp_dual_svm_ls_BB88.FlpDualLSBBSVM(lambd=1, max_iter=50, kernel="linear")
# svm_ls = flp_dual_svm_ls_scaled.FlpDualScaledLSSVM(lambd=1, lr=0.1, max_iter=50)

svm_ls = flp_dual_svm_ls.FlpDualLSSVM(lambd=20, lr=0.001, max_iter=2, kernel="linear")
time_a = datetime.datetime.now()
svm_ls.fit(X, y)
print(svm_ls.info["denominator"])

# print("Fit time SGD =", datetime.datetime.now() - time_a)
# training_score = svm_ls.score(X, y)
# print("Accuracy Train SGD =", training_score)
# X_test, y_test = load_dataset("source/experiments/real_experiment/datasets/dataset_test_0.csv")
# test_score = svm_ls.score(X_test, y_test)
# print("Accuracy Test SGD =", test_score)



# prediction = svm_ls.predict(X)

# fig, axs = plt.subplots(2, 2)

# axs[0, 0].plot(svm.info["accuracy"])
# axs[0, 0].set_title("Accuracy")

# axs[0, 1].plot(svm.info["pk_norm"], color='blue', lw=2)
# axs[0, 1].set_yscale("log")
# axs[0, 1].set_title("Pk norm")

# axs[0, 0].scatter(X[:,0], X[:,1], c=y.reshape(-1), cmap='viridis')
# axs[0, 0].set_title("Real dataset")

# axs[0, 1].scatter(X[:,0], X[:,1], c=prediction.reshape(-1), cmap='viridis')
# axs[0, 1].set_title("Predictions")

# plt.show() 
