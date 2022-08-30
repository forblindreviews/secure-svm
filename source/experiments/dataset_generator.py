from sklearn import datasets
import numpy as np
import pandas as pd
import random
import json
from sklearn.preprocessing import MinMaxScaler


# Set seeds for RNGs
np.random.seed(1)
random.seed(1)


# Load information
with open("source/experiments/config.json") as config_file:
    config = json.load(config_file)
with open("source/experiments/experiment_info.json") as file_info:
    data_experiments = json.load(file_info)


def generate_dataset(n_samples, n_features, class_sep):
    X, y = datasets.make_classification(
        n_samples,
        n_features,
        n_redundant=0,
        n_repeated=0,
        n_informative=n_features,
        class_sep=class_sep
    )
    y = pd.Series(y).map({0: -1, 1: 1}).values

    # Extend y columns
    y = np.expand_dims(y, axis=1)
    
    # Scale dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def split_dataset(X, y, train_percentage):
    size_train = int(train_percentage * X.shape[0])

    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:size_train], indices[size_train:]
    X_train, X_test = X[training_idx,:], X[test_idx,:]
    y_train, y_test = y[training_idx,:], y[test_idx,:]
    return X_train, X_test, y_train, y_test


def save_dataset_csv(X, y, experiment, n_execution, label):
    df_save = pd.DataFrame(data=np.append(X, y, axis=1))
    file_name = "toy_dataset_" + label + "_" + str(n_execution) + ".csv"
    df_save.to_csv(config["experiments_path"] + experiment + "/datasets/" + file_name, index=False, columns=None)


def save_dataset_parties(X, y, n_parties):
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    rows_per_party = n_rows // n_parties
    last_party = 0 
    if n_rows % n_parties != 0:
        last_party = rows_per_party + (n_rows % n_parties)
    else:
        last_party = rows_per_party
    
    party_info_X = []
    party_info_y = []
    for i in range(n_parties - 1):
        party_X_rows = []
        party_y_rows = []
        for j in range(rows_per_party):
            party_X_rows.append(X[j + i * rows_per_party].tolist())
            party_y_rows.append(y[j + i * rows_per_party][0])
        party_info_X.append(party_X_rows)
        party_info_y.append(party_y_rows)

    # Last party
    party_X_rows = []
    party_y_rows = []
    for j in range(last_party):
        party_X_rows.append(X[j + rows_per_party * (n_parties - 1)].tolist())
        party_y_rows.append(y[j + rows_per_party * (n_parties - 1)][0])
    party_info_X.append(party_X_rows)
    party_info_y.append(party_y_rows)

    for i in range(n_parties - 1):
        file_name = config["mp_spdz_path"] + "Player-Data/Input-P" + str(i) + "-0"
        file = open(file_name, "w")
        file_str = ""
        for j in range(rows_per_party):
            for k in range(n_cols):
                file_str += str(party_info_X[i][j][k]) + " "
            file_str = file_str.strip()
            file_str += "\n"
        
        for j in range(rows_per_party):
            file_str += str(party_info_y[i][j]) + "\n"
        
        file.write(file_str)
        file.close()
    
    # Last party write
    file_name = config["mp_spdz_path"] + "Player-Data/Input-P" + str(n_parties - 1) + "-0"
    file = open(file_name, "w")
    file_str = ""
    for j in range(last_party):
        for k in range(n_cols):
            file_str += str(party_info_X[n_parties - 1][j][k]) + " "
        file_str = file_str.strip()
        file_str += "\n"
    
    for j in range(last_party):
        file_str += str(party_info_y[n_parties - 1][j]) + "\n"
    
    file.write(file_str)
    file.close()
