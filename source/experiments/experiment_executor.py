
from fileinput import filename
from operator import index
import subprocess
import json
import math
import os
import pandas as pd
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(""))

import source.experiments.dataset_generator as dataset_generator
import source.analyzers.analyzer_utils as analyzer_utils

# Load information
with open("source/experiments/config.json") as config_file:
    config = json.load(config_file)
with open("source/experiments/experiment_info.json") as file_info:
    data_experiments = json.load(file_info)

# File for error logging
file_error = open(config["log_file"], "w")


def compute_ring_size(f, k):
    """
    Compute the ring size given the precision.

    param f: precision for integer part.
    param k: total number of bits in the whole word.
    """

    f_new = max(k // 2 + 1, f)
    ring_size = max(2 * k + 1, k + 3 * f_new - f)
    return ring_size


def compute_precision(experiment_name, algorithm):
    """
    Computes the precision needed for this experiment

    param experiment_name: name of the experiment to be executed.
    """

    n_rows = data_experiments[experiment_name]["n_rows"]
    n_cols = data_experiments[experiment_name]["n_columns"]

    if algorithm["name"] == "ls" or algorithm["name"] == "smo" or algorithm["name"] == "sgd":
        integer_precision = math.floor(
            math.log2((n_rows ** 7) * (n_cols ** 6))) + 1
    elif algorithm["name"] == "scaled":
        integer_precision = math.floor(
            math.log2((n_rows ** 5) * (n_cols ** 3))) + 1
    elif algorithm["name"] == "bb88":
        integer_precision = math.floor(
            math.log2((n_rows ** 5) * (n_cols ** 4))) + 1
    elif algorithm["name"] == "logreg":
        integer_precision = math.floor(
            math.log2(n_rows)) + 1

    # Computes the fractional precision
    f = math.ceil(math.log2((n_rows ** 2) * (n_cols ** 2))) + 1

    # Computes the word length    
    k = integer_precision + f
    
    return f, k


def execute_real_experiment(experiment_name):
    
    print("==========================> RUNNING EXPRIMENT",
          experiment_name, "<==========================")

    if not os.path.exists('source/experiments/' + experiment_name):
        os.makedirs('source/experiments/' + experiment_name)

    path_dataset_train = "source/experiments/real_experiment/datasets/" + \
        data_experiments[experiment_name]["dataset"] + "_train_0.csv"
    complete_df_train = pd.read_csv(path_dataset_train)
    X_train = complete_df_train.iloc[:, 0:complete_df_train.shape[1] - 1].to_numpy()
    y_train = complete_df_train.iloc[:, complete_df_train.shape[1] - 1].to_numpy()
    y_train = np.expand_dims(y_train, axis=1)
    
    # Response variable for logreg
    y_train_logreg = y_train.copy()
    y_train_logreg[y_train_logreg == -1] = 0

    for protocol in data_experiments[experiment_name]["protocols"]:
        print("==> Executing protocol", protocol["name"])

        for algorithm in data_experiments[experiment_name]["algorithms"]:
            # Save dataset for MP-SPDZ
            if algorithm["name"] == "logreg":
                dataset_generator.save_dataset_parties(X_train, y_train_logreg, protocol["n_parties"])
            else:
                dataset_generator.save_dataset_parties(X_train, y_train, protocol["n_parties"])
            
            print("====> Executing algorithm", algorithm["name"])

            # Compile the library of the given protocol and algorithm
            compile_library(protocol, experiment_name, algorithm)

            # Compile the bytecode of the algorithm for the given protocol
            compile_bytecode(experiment_name, algorithm, protocol)

            for repetition in range(data_experiments[experiment_name]["n_repetitions"]):
                print("========> Executing repetition", repetition)
                
                result_str = execute_secure_algorithm(
                    experiment_name, algorithm, protocol)
                save_results(result_str, experiment_name,
                                algorithm, protocol, repetition)
    


def execute_artificial_experiment(experiment_name):
    """
    Execute experiments that are not model selection.

    param experiment_name: name of the experiment being executed.
    """

    print("==========================> RUNNING EXPRIMENT",
          experiment_name, "<==========================")

    # Prepare directories
    if not os.path.exists('source/experiments/' + experiment_name):
        os.makedirs('source/experiments/' + experiment_name)
    if not os.path.exists('source/experiments/' + experiment_name + "/datasets"):
        os.makedirs('source/experiments/' + experiment_name + "/datasets")

    first_dataset = False
    for protocol in data_experiments[experiment_name]["protocols"]:
        print("==> Executing protocol", protocol["name"])

        for repetition in range(data_experiments[experiment_name]["n_repetitions"]):
            print("========> Executing repetition", repetition)
            if not first_dataset or (data_experiments[experiment_name]["change_dataset"]):
                generate_dataset_experiment(
                    experiment_name, protocol, repetition)
                first_dataset = True

            # Correct the input files of the first dataset for MP-SPDZ protocol to allow a different number of parties
            if first_dataset and not data_experiments[experiment_name]["change_dataset"]:
                print("Updating input files to work with the current protocol...")
                path_train = "source/experiments/" + experiment_name + \
                    "/datasets/" + "toy_dataset_train_0.csv"
                X_train, y_train = analyzer_utils.load_dataset(path_train)
                dataset_generator.save_dataset_parties(
                    X_train, y_train, protocol["n_parties"])

            for algorithm in data_experiments[experiment_name]["algorithms"]:
                print("====> Executing algorithm", algorithm["name"])
                # Compile the library of the given protocol
                compile_library(protocol, experiment_name, algorithm)

                # Compile the bytecode of the algorithm for the given protocol
                compile_bytecode(experiment_name, algorithm, protocol)

                result_str = execute_secure_algorithm(
                    experiment_name, algorithm, protocol)
                save_results(result_str, experiment_name,
                             algorithm, protocol, repetition)


def execute_secure_algorithm(experiment_name, algorithm, protocol):
    """
    Executes the secure algorithm in MP-SPDZ

    param algorithm: dictionary with the information of the ML algorithm.
    param protocol: dictionary with the information of the MPC protocol to be executed.
    """

    # Generate execution command
    cd_command = "cd " + config["mp_spdz_path"]
    algorithm_src_name = algorithm["script"]

    compilation_params = extract_compilation_params(
        experiment_name, algorithm, protocol)
    exec_command = "Scripts/" + protocol["script"] + " --verbose " + \
        algorithm_src_name + "-" + \
        "-".join([str(param) for param in compilation_params])
    command = cd_command + " && " + exec_command

    # Execute the command
    print("Running:", command)
    result = subprocess.run(
        [command],
        stdout=subprocess.PIPE,
        shell=True,
        stderr=file_error
    )

    result.check_returncode()
    result_str = result.stdout.decode('utf-8')
    return result_str


def compile_library(protocol, experiment_name, algorithm):
    """
    Recompiles the library to allow computations in the required ring size.

    param protocol: a dictionary with the information of the protocol.
    param experiment_name: name of the experiment for wich the library will be compiled.
    """

    # CONFIG.mine file content
    str_tail_config_mpspdz_file = (
        "MY_CFLAGS += -I./local/include\n"
        "MY_LDLIBS += -Wl,-rpath -Wl,./local/lib -L./local/lib\n"
        "ARCH = -march=native\n"
    )
    f, k = compute_precision(experiment_name, algorithm)
    ring_size = compute_ring_size(f, k)
    mod_ring_size_command = "MOD = -DRING_SIZE=" + str(ring_size)
    config_mpspdz_file_content = mod_ring_size_command + \
        "\n\n" + str_tail_config_mpspdz_file

    # Save CONFIG.mine file
    print("Saving CONFIG.mine file...")
    config_file = open(config["mp_spdz_path"] + "CONFIG.mine", "w")
    config_file.write(config_mpspdz_file_content)
    config_file.close()

    compile_command = "make clean && make -j 8 " + protocol["name"] + "-party.x"

    # Compiles the library for the specified ring size.
    compile_command = "cd " + config["mp_spdz_path"] + " && " + compile_command
    print("Running:", compile_command)
    result = subprocess.run(
        [compile_command],
        stdout=subprocess.PIPE,
        shell=True,
        stderr=file_error
    )
    result.check_returncode()


def compile_bytecode(experiment_name, algorithm, protocol):
    """
    Compiles the bytecode of the secure LS SVM to execute in MP-SPDZ.

    param experiment_name: experiment in wich the algorithm will be compiled.
    param algorithm: algorithm whose source code will be compiled.
    param protocol: protocol for wich the algorithm will be compiled
    """

    algorithm_src_file = algorithm["script"] + ".mpc"

    # Copies the source code from the experiment folder to the MP-SPDZ folder.
    if algorithm["name"] == "logreg":
        copy_command = "cp -rf " + config["secure_logreg_path"] + algorithm_src_file + \
            " " + config["mp_spdz_path"] + "Programs/Source/" + algorithm_src_file    
    else:
        copy_command = "cp -rf " + config["secure_src_path"] + algorithm_src_file + \
            " " + config["mp_spdz_path"] + "Programs/Source/" + algorithm_src_file

    print("Running:", copy_command)
    result = subprocess.run(
        [copy_command],
        stdout=subprocess.PIPE,
        shell=True,
        stderr=file_error
    )
    result.check_returncode()

    compilation_params = extract_compilation_params(
        experiment_name, algorithm, protocol)
    f = compilation_params[3]
    k = compilation_params[4]
    compile_command = "./compile.py -R " + \
        str(compute_ring_size(f, k)) + " " + \
        algorithm_src_file + " " + \
        " ".join([str(param) for param in compilation_params])

    final_compile_command = "cd " + \
        config["mp_spdz_path"] + " && " + compile_command
    print("Running:", final_compile_command)
    result = subprocess.run(
        [final_compile_command],
        stdout=subprocess.PIPE,
        shell=True,
        stderr=file_error
    )
    result.check_returncode()


def extract_compilation_params(experiment_name, algorithm, protocol):
    """
    Extract the compilation params to compile an algorithm source code in MP-SPDZ

    param experiment_name: experiment in wich the algorithm will be compiled.
    param algorithm: algorithm whose source code will be compiled.
    param protocol: protocol for wich the algorithm will be compiled
    """

    f, k = compute_precision(experiment_name, algorithm)
    compilation_params = [
        data_experiments[experiment_name]["n_rows"],
        data_experiments[experiment_name]["n_columns"],
        protocol["n_parties"],
        f,
        k
    ]

    compilation_params = compilation_params + \
        list(algorithm["params"].values())

    return compilation_params


def save_results(result_str, experiment_name, algorithm, protocol, repetition):
    """
    Saves the experiment output in a file.

    param result_str: string that contains the otput of the execution.
    param experiment_name: name of the executed experiment.
    param algorithm: dictionary with the information of the algorithm executed
    param protocol: dictionary with the information of the protocol executed
    param repetition: index of the repetition of the experiment
    """

    
    file_name = "ouput_secure_" + \
        algorithm["name"] + "_" + protocol["name"] + \
        "_" + str(repetition) + ".txt"
    path = config["experiments_path"] + experiment_name + "/"
    print("Saving results in", path + file_name)
    file_output = open(path + file_name, "w")
    file_output.write(result_str)
    file_output.close()


def generate_dataset_experiment(experiment_name, protocol, repetition):
    """
    Generates de dataset for the specified experiment.

    param experiment_name: name of the experiment for wich we are generating the dataset.
    param protocol: dictionary with the protocol information.
    param repetition: index of the current repetition of the experiment
    """

    print("Generating dataset...")
    X, y = dataset_generator.generate_dataset(
        math.ceil(data_experiments[experiment_name]["n_rows"] /
                  data_experiments[experiment_name]["train_percentage"]),
        data_experiments[experiment_name]["n_columns"],
        data_experiments[experiment_name]["class_sep"]
    )

    X_train, X_test, y_train, y_test = dataset_generator.split_dataset(
        X, y, data_experiments[experiment_name]["train_percentage"]
    )

    dataset_generator.save_dataset_csv(
        X_train, y_train, experiment_name, repetition, "train")
    dataset_generator.save_dataset_csv(
        X_test, y_test, experiment_name, repetition, "test")
    dataset_generator.save_dataset_csv(
        X, y, experiment_name, repetition, "complete")

    dataset_generator.save_dataset_parties(
        X_train, y_train, protocol["n_parties"])


if __name__ == "__main__":
    experiment_list = [
        # "model_selection_no_class_sep",
        # "model_selection_05",
        # "model_selection_10",

        # "test_100r_4c",
        # "test_150r_4c",
        # "test_200r_4c",
        # "test_250r_4c",
        # "test_300r_4c",
        # "test_350r_4c",
        # "test_400r_4c",
        
        "test_400r_2c",
        "test_400r_3c",
        "test_400r_5c",
        "test_400r_6c",
        "test_400r_7c",
        "test_400r_8c"

        # "test_class_sep_040",
        # "test_class_sep_050",
        # "test_class_sep_060",
        # "test_class_sep_070",
        # "test_class_sep_080",
        # "test_class_sep_090",
        # "test_class_sep_100",
        # "test_class_sep_110",
        # "test_class_sep_120",

        # "ls_selection_100r",
        # "ls_selection_200r",
        # "ls_selection_300r",
        # "ls_selection_400r"

        # "real_experiment_400"
        
        # "real_experiment_hcc",
        # "real_experiment_credit",
        
        # "network_experiment",
        # "real_experiment_definitive"
    ]

    for experiment in experiment_list:
        if experiment.startswith("real_experiment"):
            execute_real_experiment(experiment)
        else:
            execute_artificial_experiment(experiment)

    file_error.close()
