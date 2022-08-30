from fxp_svm_ls import FxpDualLSSVM
import numpy as np
from fxpmath import Fxp
from sklearn.datasets import make_classification
import pandas as pd
import math
import sys
import os

from pytest import param
sys.path.insert(0, os.path.abspath(""))

from source.floating_svm.flp_dual_svm_ls import FlpDualLSSVM

# Generate dataset
# X, y = make_classification(n_samples=40, n_features=5, n_redundant=0, n_informative=2)
# y = pd.Series(y).map({0: -1, 1: 1}).values
# y = np.expand_dims(y, axis=1)

# Load dataset train
df_train = pd.read_csv("source/experiments/real_experiment/datasets/dataset_train_0.csv")
X_train = df_train.iloc[:, :df_train.shape[1] - 1]
y_train = df_train.iloc[:, df_train.shape[1] - 1]

# Load dataset test
df_test = pd.read_csv("source/experiments/real_experiment/datasets/dataset_test_0.csv")
X_test = df_test.iloc[:, :df_test.shape[1] - 1]
y_test = df_test.iloc[:, df_test.shape[1] - 1]

y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

# Computation of integer precision
integer_precision = math.floor(
            math.log2((X_train.shape[0] ** 7) * (X_train.shape[1] ** 6))) + 1

# Fractional precision (length to fractional precision)
f = math.ceil(math.log2((X_train.shape[0] ** 2) * (X_train.shape[1] ** 2))) + 1

# Word length
k = integer_precision + f

print("==============================")
print("==> FIXED-POINT RESULTS <==")
print("Length word:", k)
print("Length frac:", f)
fxp_svm = FxpDualLSSVM(lambd=10, lr=0.1, max_iter=400, length_word=k, length_frac=f)
fxp_svm.fit(X_train.to_numpy(), y_train)
print("Score train:", fxp_svm.score(X_train.to_numpy(), y_train))
print("Score test:", fxp_svm.score(X_test.to_numpy(), y_test))
print("==============================")
print("==> FLOATING-POINT RESULTS <==")
flp_svm = FlpDualLSSVM(lambd=10, lr=0.1, max_iter=400)
flp_svm.fit(X_train.to_numpy(), y_train)
print("Score train:", flp_svm.score(X_train.to_numpy(), y_train))
print("Score test:", flp_svm.score(X_test.to_numpy(), y_test))