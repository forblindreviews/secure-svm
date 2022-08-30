from os import path, sep
import numpy as np
from numpy.core.records import array
import pandas as pd
import datetime
import json


# Add parent project directory to path
import os, sys
sys.path.insert(0, os.path.abspath(""))

import source.floating_svm.flp_dual_svm as flp_dual_svm
import source.floating_svm.flp_dual_svm_ls as flp_dual_svm_ls
import source.floating_svm.flp_svm as flp_svm

from analyzer_utils import *


with open("source/experiments/config.json") as config_file:
    config = json.load(config_file)
with open("source/experiments/experiment_info.json") as file_info:
    data_experiments = json.load(file_info)


if __name__ == "__main__":
    experiment_list = [
        "test-40r-2c",
        "test-50r-2c",
        "test-60r-2c",
        "test-70r-2c",
        "test-80r-2c",
        "test-90r-2c",

        "test-100r-2c",
        "test-100r-3c",
        "test-100r-4c",
        "test-100r-5c",
        "test-100r-6c",
        "test-100r-7c",
        "test-100r-8c",
        "test-100r-9c",
        "test-100r-10c",

        "test_class_sep_040",
        "test_class_sep_050",
        "test_class_sep_060",
        "test_class_sep_070",
        "test_class_sep_080",
        "test_class_sep_090",
        "test_class_sep_100",
        "test_class_sep_110"
    ]

    algorithm = "ls"

    result_data = []
    for experiment in experiment_list:
        first_dataset = False
        

        for n_execution in range(data_experiments[experiment]["n_executions"]):
            if not first_dataset or data_experiments[experiment]["change_dataset"]:
                dataset_name = "toy_dataset"
                path_train = config["experiments_path"] + experiment + "/datasets/" + dataset_name + "_train_" + str(n_execution) + ".csv"
                path_test = config["experiments_path"] + experiment + "/datasets/" + dataset_name + "_test_" + str(n_execution) + ".csv"
                first_dataset = True
            
            path_parameters = config["experiments_path"] + experiment + "/ouput_secure_" + str(n_execution) + ".txt"
            
            X_train, y_train = load_dataset(path_train)
            X_test, y_test = load_dataset(path_test)

            parameters = get_parameters(path_parameters)
            sec_time, data_sent, global_data_sent = get_execution_info(path_parameters)

            model = load_parameters(parameters, algorithm, X_train, y_train)
            print("#############", experiment, "-", n_execution, "#############")
            print("===> Secure SVM")
            sec_train_score = model.score(X_train, y_train)
            sec_test_score = model.score(X_test, y_test)
            print("Train acc =", sec_train_score)
            print("Test acc =", sec_test_score)

            print("===> Traditional SVM")
            time_a = datetime.datetime.now()
            model.fit(X_train, y_train)
            clean_time_date = datetime.datetime.now() - time_a
            clean_time = float(str(clean_time_date.seconds) + "." + str(clean_time_date.microseconds))
            print("Fit time =", clean_time)
            clean_train_score = model.score(X_train, y_train)
            print("Training accuracy =", clean_train_score)
            clean_test_score = model.score(X_test, y_test)
            print("Test accuracy =", clean_test_score)


            experiment_results = [
                data_experiments[experiment]["n_rows"],
                data_experiments[experiment]["n_columns"],
                data_experiments[experiment]["class_sep"],
                n_execution,
                sec_train_score,
                sec_test_score,
                clean_train_score,
                clean_test_score,
                sec_time,
                clean_time,
                data_sent,
                global_data_sent
            ]
                    
            result_data.append(experiment_results)
    

    headers = [
        "Rows",
        "Columns",
        "Class sep",
        "N execution",
        "Train acc sec",
        "Test acc sec",
        "Train acc clean",
        "Test acc clean",
        "Time sec",
        "Time clean",
        "Data sent",
        "Global data sent",
    ]

    df_results = pd.DataFrame(
        data=np.array(result_data),
        columns=headers
    )

    df_results.to_csv(config["experiments_path"] + "results_repeated.csv")
    