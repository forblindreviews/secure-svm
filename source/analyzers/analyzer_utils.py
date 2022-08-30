from os import path, sep
import numpy as np
from numpy.core.records import array
import pandas as pd
import datetime
import json

# Add parent project directory to path
import os, sys

from pytest import param
sys.path.insert(0, os.path.abspath(""))

import source.floating_svm.flp_dual_svm as flp_dual_svm
import source.floating_svm.flp_dual_svm_ls as flp_dual_svm_ls
import source.floating_svm.flp_svm as flp_svm
import source.floating_svm.flp_dual_svm_ls_BB88 as flp_dual_svm_ls_BB88
import source.floating_svm.flp_dual_svm_ls_scaled as flp_dual_svm_ls_scaled
import source.floating_logistic_regression.flp_logistic as flp_logistic


def get_model_clear_with_params(algorithm_info):
    params = algorithm_info["params"]
    if algorithm_info["name"] == "ls":
        model = flp_dual_svm_ls.FlpDualLSSVM(
            lambd=params["lambd"],
            lr=params["lr"],
            max_iter=params["max_iter"]
        )
    elif algorithm_info["name"] == "smo":
        model = flp_dual_svm.FlpDualSVM(
            C=params["C"],
            eps=params["eps"],
            tolerance=params["tolerance"],
            max_phases=params["max_phases"]
        )
    elif algorithm_info["name"] == "sgd":
        model = flp_svm.FlpSVM(
            lambd=params["lambd"],
            lr=params["lr"],
            epochs=params["epochs"]
        )
    elif algorithm_info["name"] == "scaled":
        model = flp_dual_svm_ls_scaled.FlpDualScaledLSSVM(
            lambd=params["lambd"],
            lr=params["lr"],
            max_iter=params["max_iter"]
        )
    elif algorithm_info["name"] == "bb88":
        model = flp_dual_svm_ls_BB88.FlpDualLSBBSVM(
            lambd=params["lambd"],
            max_iter=params["max_iter"]
        )
    elif algorithm_info["name"] == "logreg":
        model = flp_logistic.LogisticRegression(
            lr=params["lr"],
            threshold=params["threshold"],
            max_iter=params["max_iter"]
        )

    return model


def load_parameters(parameters, algorithm, X_train, y_train):
    params_list = list()
    for line in parameters.split("\n"):
        param_data = float(line.strip("\n").strip("[").strip("]"))
        params_list.append([param_data])

    params = np.array(params_list)
    
    if algorithm["name"] == "sgd":
        model = flp_svm.FlpSVM()
        W = params[0:(params.shape[0] - 1)]
        b = params[len(params) - 1][0]
        model.load_parameters(W, b)
    elif algorithm["name"] == "smo":
        model = flp_dual_svm.FlpDualSVM()
        alphas = params[:params.shape[0] - 1]
        b = params[params.shape[0] - 1][0]
        model.load_parameters(alphas, b, X_train, y_train)
    elif algorithm["name"] == "ls":
        model = flp_dual_svm_ls.FlpDualLSSVM()
        alphas = params[:params.shape[0] - 1]
        b = params[params.shape[0] - 1][0]
        model.load_parameters(alphas, b, X_train, y_train)
    elif algorithm["name"] == "bb88":
        model = flp_dual_svm_ls_BB88.FlpDualLSBBSVM()
        alphas = params[:params.shape[0] - 1]
        b = params[params.shape[0] - 1][0]
        model.load_parameters(alphas, b, X_train, y_train)
    elif algorithm["name"] == "scaled":
        model = flp_dual_svm_ls_scaled.FlpDualScaledLSSVM()
        alphas = params[:params.shape[0] - 1]
        b = params[params.shape[0] - 1][0]
        model.load_parameters(alphas, b, X_train, y_train)
    elif algorithm["name"] == "logreg":
        model = flp_logistic.LogisticRegression()
        model.load_parameters(params, algorithm["params"]["threshold"])
    return model


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :df.shape[1] - 1]
    y = df.iloc[:, df.shape[1] - 1]

    y = np.expand_dims(y, axis=1)

    return X.to_numpy(), y


def get_parameters(path_parameters):
    file_parameters = open(path_parameters, "r")
    file_content = file_parameters.read()
    parts_file_content = file_content.split("--\n")
    parameters = parts_file_content[1].strip("\n")
    file_parameters.close()
    return parameters


def get_execution_info(path_parameters):
    file_parameters = open(path_parameters, "r")
    file_content = file_parameters.read()
    file_parts = file_content.split("--\n")
    execution_info_part = file_parts[2].strip("\n")
    execution_info_splitted = execution_info_part.split("\n")

    for line in execution_info_splitted:
        if line.startswith("Time ="):
            time = float(line.lstrip("Time = ").rstrip(" seconds"))
        elif line.startswith("Data sent ="):
            data = float(line.lstrip("Data sent = ").split(" MB ")[0])
        elif line.startswith("Global data sent ="):
            global_data = float(line.lstrip("Global data sent = ").split(" MB ")[0])

    file_parameters.close()

    return time, data, global_data