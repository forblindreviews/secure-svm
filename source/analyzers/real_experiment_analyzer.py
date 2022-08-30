from os import name, path, sep
import numpy as np
from numpy.core.records import array
from numpy.lib.npyio import load
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
    # Real experiment
    experiment = "real_experiment"
    dataset_name = "real_dataset"
        
    path_train = "source/experiments/" + experiment + "/datasets/" + dataset_name + "_train.csv"
    path_test = "source/experiments/" + experiment + "/datasets/" + dataset_name + "_test.csv"

    path_parameters = "source/experiments/" + experiment + "/svm_ls_results.txt"
    algorithm = "ls"

    X_train, y_train = load_dataset(path_train)
    X_test, y_test = load_dataset(path_test)

    parameters = get_parameters(path_parameters)
    sec_time, data_sent, global_data_sent = get_execution_info(path_parameters)
    
    model = load_parameters(parameters, algorithm, X_train, y_train)
    print("#############", experiment, "#############")
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