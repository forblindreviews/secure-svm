import os, sys

from sklearn import datasets
sys.path.insert(0, os.path.abspath(""))

import source.experiments.dataset_generator as data
import pandas as pd
import numpy as np 

def generate_dataset(n_samples, n_features):
    X, y = datasets.make_classification(n_samples, n_features, n_redundant=0, n_informative=2)
    y = pd.Series(y).map({0: -1, 1: 1}).values

    # Extend y columns
    y = np.expand_dims(y, axis=1)

    # Save dataset for MATLAB testing
    # df_save = pd.DataFrame(data=np.append(X, y, axis=1))
    # df_save.to_csv("source/datasets/toy_dataset.csv", index=False, columns=None)

    return X, y

def load_train_dataset_from_file(path):
    df_train = pd.read_csv("source/experiments/real_experiment/datasets/dataset_train_0.csv")
    X_train = df_train.iloc[:, :df_train.shape[1] - 1]
    y_train = df_train.iloc[:, df_train.shape[1] - 1]
    y_train = np.expand_dims(y_train, axis=1)
    return X_train.to_numpy(), y_train


# Generate random artificial dataset
# X_train, y_train = generate_dataset(4, 2)

# Load dataset from path
path = "source/experiments/real_experiment/datasets/dataset_test_0.csv"
X_train, y_train = load_train_dataset_from_file(path)
print(X_train.shape)

data.save_dataset_parties(X_train, y_train, 1)
