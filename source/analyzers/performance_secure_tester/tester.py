import os, sys
from statistics import mode
sys.path.insert(0, os.path.abspath(""))

from source.analyzers.analyzer_utils import *
from source.floating_svm.flp_dual_svm_ls import FlpDualLSSVM

from numpy.core.records import array


if __name__ == "__main__":
    path_parameters = "source/analyzers/performance_secure_tester/parameters.txt"
    path_train = "source/analyzers/performance_secure_tester/train.csv"
    path_test = "source/analyzers/performance_secure_tester/test.csv"

    parameters = open(path_parameters, "r").read()
    X_train, y_train = load_dataset(path_train)
    X_test, y_test = load_dataset(path_test)

    model = load_parameters(parameters, algorithm="ls", X_train=X_train, y_train=y_train)

    print("##########################")
    print("===> Secure SVM")
    sec_train_score = model.score(X_train, y_train)
    sec_test_score = model.score(X_test, y_test)
    print("Train acc =", sec_train_score)
    print("Test acc =", sec_test_score)

    print("===> Floating SVM")
    model_flp = FlpDualLSSVM(lambd=0.1, lr=0.1, max_iter=400)
    model_flp.fit(X_train, y_train)
    flp_train_score = model_flp.score(X_train, y_train)
    flp_test_score = model_flp.score(X_test, y_test)
    print("Train acc =", flp_train_score)
    print("Test acc =", flp_test_score)
