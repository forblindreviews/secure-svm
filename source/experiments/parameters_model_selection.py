import os, sys

from pytest import param
sys.path.insert(0, os.path.abspath(""))

import source.floating_svm.flp_dual_svm as flp_dual_svm
import source.floating_svm.flp_dual_svm_ls as flp_dual_svm_ls
import source.floating_svm.flp_svm as flp_svm
import source.experiments.dataset_generator as dataset_generator

from sklearn.model_selection import train_test_split


def test_ls_algorithm(params, X_train, y_train, X_test, y_test):
    print("============ Testing LS =====================")
    max_accuracy = 0
    best_params = {}
    for lr in params["lr"]:
        for lambd in params["lambd"]:
            print("==> Testing", "lr = ", lr, "lambd =", lambd)
            model = flp_dual_svm_ls.FlpDualLSSVM(lambd=lambd, lr=lr, max_iter=params["max_iter"])
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_params["lr"] = lr
                best_params["lambd"] = lambd
    
    return {"max_accuracy": max_accuracy, "best_params": best_params}


def test_smo_algorithm(params, X_train, y_train, X_test, y_test):
    print("============ Testing SMO =====================")
    max_accuracy = 0
    best_params = {}
    for C in params["C"]:
        for tolerance in params["tolerance"]:
            for eps in params["eps"]:
                print("==> Testing", "C = ", C, ", tolerance =", tolerance, ", eps =", eps)
                model = flp_dual_svm.FlpDualSVM(
                    C=C,
                    eps=eps,
                    tolerance=tolerance,
                    max_phases=params["max_phases"]
                )
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_params["C"] = C
                    best_params["tolerance"] = tolerance
                    best_params["eps"] = eps

    return {"max_accuracy": max_accuracy, "best_params": best_params}


def test_sgd_algorithm(params, X_train, y_train, X_test, y_test):
    print("============ Testing SGD =====================")
    max_accuracy = 0
    best_params = {}
    for lr in params["lr"]:
        for lambd in params["lambd"]:
            print("==> Testing", "lr = ", lr, "lambd =", lambd)
            model = flp_svm.FlpSVM(
                lambd=lambd,
                lr=lr,
                epochs=params["epochs"]
            )
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_params["lr"] = lr
                best_params["lambd"] = lambd
    
    return {"max_accuracy": max_accuracy, "best_params": best_params}



if __name__ == "__main__":
    n_samples = 100
    n_columns = 5
    X, y = dataset_generator.generate_dataset(n_samples=n_samples * 2, n_features=n_columns, class_sep=1.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    ls_params_test = {
        "lambd": [0.01, 0.1, 1, 10, 20, 50, 100],
        "lr": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        "max_iter": 100
    }

    sgd_params_test = {
        "lr": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        "lambd": [0.1, 1, 2, 4, 8, 16, 32, 64, 128],
        "epochs": 100
    } 

    smo_params_test = {
        "C": [0.1, 1, 2, 4, 8, 16, 32, 64, 128],
        "tolerance": [1e-4, 1e-3, 1e-2, 0.1], 
        "eps": [1e-4, 1e-3, 1e-2, 0.1],
        "max_phases": 100
    } 

    # results_ls = test_ls_algorithm(ls_params_test, X_train, y_train, X_test, y_test)
    # results_smo = test_smo_algorithm(smo_params_test, X_train, y_train, X_test, y_test)
    results_sgd = test_sgd_algorithm(sgd_params_test, X_train, y_train, X_test, y_test)

    print("Results SGD:", results_sgd)
    #print("Results SMO:", results_smo)
    #print("Results LS:", results_ls)

    #file_params = open("source/experiments/best_parameters_model_selection.txt", "w")
    #file_params.write(str(results_sgd) + "\n" + str(results_smo) + "\n" + str(results_ls))
    #file_params.close()