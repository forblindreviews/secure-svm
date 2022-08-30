import os, sys

from pytest import param
sys.path.insert(0, os.path.abspath(""))

import source.floating_svm.flp_dual_svm as flp_dual_svm
import source.floating_svm.flp_dual_svm_ls as flp_dual_svm_ls
import source.floating_svm.flp_svm as flp_svm
import source.floating_svm.flp_svm_sgd as flp_svm_sgd
import source.experiments.dataset_generator as dataset_generator

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Define models
    ls_model = flp_dual_svm_ls.FlpDualLSSVM(lambd=1, lr=0.01, max_iter=50)
    smo_model = flp_dual_svm.FlpDualSVM(C=1, eps=1e-3, tolerance=1e-3, max_phases=50)
    # sgd_model = flp_svm.FlpSVM(C=4, lr=0.5)
    sgd_model = flp_svm_sgd.FlpSVM(lambd=2, lr=0.001, epochs=100)

    # Generate dataset
    X, y = dataset_generator.generate_dataset(n_samples=100, n_features=5, class_sep=1.0)

    # Train models
    #ls_model.fit(X, y)
    #smo_model.fit(X, y)
    sgd_model.fit(X, y)

    # Plot LS
    # fig_ls, ax_ls = plt.subplots()
    # ax_ls.plot(ls_model.info["pk_norm"])
    # ax_ls.set_title("LS p_k norm")
    # plt.show()

    # Plot SGD
    fig_sgd, ax_sgd = plt.subplots()
    ax_sgd.plot(sgd_model.losses)
    ax_sgd.set_title("SGD losses")
    plt.show()

    print("Score:", sgd_model.score(X, y))
    
    # Max phases reached SMO
    # print("Max phases reached SMO:", smo_model.max_phases_reached)