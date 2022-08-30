import os, sys
sys.path.insert(0, os.path.abspath(""))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json 

# Load information
with open("source/experiments/config.json") as config_file:
    config = json.load(config_file)
with open("source/experiments/experiment_info.json") as file_info:
    data_experiments = json.load(file_info)
    

def save_dataset_csv(X, y, experiment, n_execution, label):
    df_save = pd.DataFrame(data=np.append(X, y, axis=1))
    file_name = "credit_curated_" + label + "_" + str(n_execution) + ".csv"
    df_save.to_csv(config["experiments_path"] + experiment + "/datasets/" + file_name, index=False, columns=None)


def load_dataset(filename):
    df = pd.read_csv("" + filename)
    X = df.iloc[:, :df.shape[1] - 1]
    y = df.iloc[:, df.shape[1] - 1]
    y = np.expand_dims(y, axis=1)
    return X.values, y


if __name__ == "__main__":
    X_cre, y_cre = load_dataset("source/experiments/real_experiment/datasets/credit_curated.csv")
    X_cre_train, X_cre_test, y_cre_train, y_cre_test = train_test_split(
        X_cre,
        y_cre,
        test_size=0.3
    )
    
    save_dataset_csv(X_cre_train, y_cre_train, "real_experiment", 0, "train")
    save_dataset_csv(X_cre_test, y_cre_test, "real_experiment", 0, "test")