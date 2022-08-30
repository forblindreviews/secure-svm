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

if __name__ == "__main__":
    algorithm = "ls"
    path_parameters = "source/experiments/model_selection/svm_ls_parameters.txt"
    path_train = "source/experiments/model_selection/datasets/toy_dataset_train.csv"
    path_test = "source/experiments/model_selection/datasets/toy_dataset_test.csv"
            
    X_train, y_train = load_dataset(path_train)
    X_test, y_test = load_dataset(path_test)

    parameters = get_parameters(path_parameters)

    model = load_parameters(parameters, algorithm, X_train, y_train)
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