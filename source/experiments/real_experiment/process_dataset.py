import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def import_dataset(path):
    complete_df = pd.read_csv(path, na_values="?", header=None)
    X = complete_df.iloc[:, 0:complete_df.shape[1] - 1]
    y = complete_df.iloc[:, complete_df.shape[1] - 1]
    y = pd.Series(y).map({0: -1, 1: 1}).values
    y = np.expand_dims(y, axis=1)

    return X, y

def select_columns(X, indexes):
    return X[:, indexes]

def remove_na(X):
    for column in range(X.shape[1]):
        if (column >= 0 and column <= 22) or (column >= 26 and column <= 28):
            X.iloc[:, column].fillna(X.iloc[:, column].mode()[0], inplace=True)
        else:
            X.iloc[:, column].fillna(X.iloc[:, column].mean(), inplace=True)
    return X


if __name__ == "__main__":
    print("Preparing dataset...")
    path_dataset = "source/experiments/real_experiment/datasets/dataset_complete_0.csv"

    X, y = import_dataset(path_dataset)
    X = remove_na(X)

    #==== For testing TODO REMOVE!!
    # X = X.sample(10, axis=0)
    # y = y[X.index]
    # print(y.shape)
    # print(X.shape)
    #====

    X = MinMaxScaler().fit_transform(X)

    # Select features according the paper
    indexes = np.nonzero([0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        0, 0, 1, 0, 1, 1, 1, 1, 1, 1
    ])[0]

    X = select_columns(X, indexes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3
    )

    # Save datasets csv
    print("Savind dataset")
    train_df = pd.DataFrame(data=np.append(X_train, y_train, axis=1))
    train_df.to_csv("source/experiments/real_experiment/datasets/dataset_train_0.csv", index=False, columns=None)
    test_df = pd.DataFrame(data=np.append(X_test, y_test, axis=1))
    test_df.to_csv("source/experiments/real_experiment/datasets/dataset_test_0.csv", index=False, columns=None)