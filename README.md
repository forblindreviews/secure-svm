# Sec-SVM

This is a repository that contains implementations of some algorithms for SVM training. There are two types of implementations: clear implementations, using Python 3 with just NumPy library, and secure implementations using MPC protocols for training which where implemented in the [Multi-Protocol SPDZ](https://github.com/data61/MP-SPDZ) framework.

## Implemented algorithms
The implemented algorithms are:
- SMO algorithm.
- Simplified SMO algorithm.
- Gradient Descent SVM optimization.
- Least squares optimization with Gradient Descent and Gauss-Seidel methods.
- Least squares optimization with [Barzilai-Borwein steep size](https://doi.org/10.1093/imanum/8.1.141).
- Least squares with a new proposal for scaling the steep size.
- Logistic regression.

The implementations are organized as follows:
- The floating point implementation (cleartext implementations) for SVM training can be found in the folder `source/floating_svm/`.
- The floating point implementation for Logistic Regression training can be found in the folder `source/secure_logistic_regression/`.
- The MP-SPDZ implementations for SVM training can be found in the folder `source/secure_svm/`.
- The MP-SPDZ implementations for Logistic Regression training can be found in the folder `source/secure_logistic_regression/`.
- The main .py file to execute the experiments is `source/experiments/experiment_executor.py`. This file executes the experiments according to the configuration file `source/experiments/experiment_info.json` whose structure will be explained below.
- The analyzers takes all the raw outputs (which are the console outputs from MP-SPDZ) after the experiment execution and organizes the results in a .csv file for further analysis. The main .py file to run the analyzers is `source/analyzers/experiment_analyzer.py`.

## Requirements

To execute the experiments properly, you must initialize the MP-SPDZ submodule. To do this, you must execute the following commands:

```
git submodule init
git submodule update
```

Once the submodule is initalized and cloned, you must compile the MP-SPDZ frameword according to the instructions presented in this [link](https://github.com/data61/MP-SPDZ#tldr-source-distribution).

## How to execute tests

### JSON configuration

There are two types of experiments. On the one hand, we implemented a module that executes the experiments using artificial datasets using the [Scikit-learn library](https://scikit-learn.org/stable/). On the other hand, we have a module that execute an experiment using the [Z-Alizadeh Sani Data Set](https://archive.ics.uci.edu/ml/datasets/Z-Alizadeh+Sani) and the [Statlog (German Credit Data) Data Set](http://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)).

For the artificial dataset experiment, the `source/experiments/experiment_info.json` file contains the information of each experiment to be executed. The attributes of each experiment are the following:
- `n_rows`: number of rows of the training dataset.
- `n_columns`: number of columns of the training dataset.
- `n_repetitions`: number of repetitions for this experiment.
- `class_sep`: class separation between the two classes used in the classification task.
- `train_percentage`: percentage of the training dataset with respect to the total size of the dataset.
- `change_dataset`: a boolean value. If the value is `true`, the artificial dataset will change between each repetition of the experiment. 
- `protocols`: list with protocols to be used in the experiment. Each protocol have the following attributes: 
    - `name`: name of the protocol to be used in the experiment.
    - `script`: name of the .sh file for the protocol according to the names used in MP-SPDZ. Currently, only the training with ring-based protocols is supported.
    - `n_parties`: number of parties for the protocol.
    - `type`: type of the protocol. At this moment, the experiments are implemented for protocols Mod $2^k$, so the value must be `"mod2k"`.
- `algorithms`: list with the algorithms to be used in the experiment. There are four types of algorithms available:
    - The training with hinge loss approach has the following attributes:
        - `name`: which is always set to be `"sgd"`.
        - `script`: which always has the value `"secure_sgd_svm_optim"`.
        - The `params` field is an object with the following attibutes:
            - `lr`: the larning rate.
            - `lambd`: the $\lambda$ parameter in the algorithm specification.
            - `epochs`: the number of epochs used to train the SVM.
    - The training with the SMO algorithm has the following attributes:
        - `name`: which is always set to be `"smo"`.
        - `script`: which always has the value `"secure_dual_svm_optim"`.
        - The `params` field is an object with the following attibutes:
            - `C`: the $C$ parameter in the specification.
            - `tolerance`: the $\delta$ parameter in the specification.
            - `eps`: the $\varepsilon$ parameter in the specification.
            - `max_phases`: number of complete inspections of the Lagrange multipliers. 
    - The training with the LS approach using the SDM algorithm has the following parameters:
        - `name`: which is always set to be `"ls"`.
        - `script`: which always has the value `"secure_dual_ls_svm_optim"`.
        - The `params` field is an object with the following attibutes:
            - `lr`: the larning rate.
            - `lambd`: the $\lambda$ parameter in the algorithm specification.
            - `max_iter`: number of iterations of the optimization.
    - The training with the LS approach using the Barzilai-Borwein step size has the following parameters:
        - `name`: which is always set to be `"bb88"`.
        - `script`: which always has the value `"secure_dual_ls_bb88"`.
        - The `params` field is an object with the following attibutes:
            - `lambd`: the $\lambda$ parameter in the algorithm specification.
            - `max_iter`: number of iterations of the optimization.
    - The training with the LS approach using the scaled step size has the following parameters:
        - `name`: which is always set to be `"scaled"`.
        - `script`: which always has the value `"secure_dual_ls_scaled"`.
        - The `params` field is an object with the following attibutes:
            - `lr`: the larning rate.
            - `lambd`: the $\lambda$ parameter in the algorithm specification.
            - `max_iter`: number of iterations of the optimization.

For the real-world experiment, the attributes for the `source/experiments/experiment_info.json` file are the same but instead of `traing_percentage` attribute, you must set the `test_percentage` attribute as a real number between 0 and 1. Also, the identifier of the real experiment in the JSON file must start with the string `"real_experiment"`. In order to choose which of the dataset should be used in the experiments, you must set the `dataset` attribute to `"credit_curated"` or `"dataset"` if you want to execute the experiment with the credit card risk dataset or with the HCC dataset respectively.

### Running the experiments

If you want to execute the experiments, you must fill all the information in the `source/experiments/experiment_info.json` file, then you must modify the `experiment_list` variable in the `source/experiments/experiment_executor.py` file with all the identifiers that you want to execute according to the JSON file. Finally, you must execute the following command:

```
python3 source/experiments/experiment_executor.py
```

In the JSON file you can find the specifications for all the experiments executed in the paper "Privacy-Preserving Machine Lerning for Support Vector Machines". Feel free to modify it and experiment with the parameters.

When the experiment execution finishes, you can generate a CSV report with the results. First you must modify the `experiment_list` variable in the `source/analyzers/experiment_analyzer.py` file with the identifiers in the JSON file which will appear in the report. Then you execute the command

```
python3 source/analyzers/experiment_analyzer.py
```

Once you have executed the analyzer, the script will generate a .csv file with the reports of all the experiments in the folder `source/experiments/reports`. For additional information about the execution of the experiments, you can find the console output of the MP-SPDZ scripts for each experiment in the folder `source/experiments/` divided in folders for each experiment. The name of the folder for each experiment will be the same as the identifier of the experiment in the JSON file.