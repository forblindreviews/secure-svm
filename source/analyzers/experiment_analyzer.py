from email import header
import os, sys
sys.path.insert(0, os.path.abspath(""))


from source.experiments.experiment_executor import compute_precision, compute_ring_size


import json
import analyzer_utils
import datetime
import pandas as pd
import numpy as np


# Load information
with open("source/experiments/config.json") as config_file:
    config = json.load(config_file)
with open("source/experiments/experiment_info.json") as file_info:
    data_experiments = json.load(file_info)


def extract_results(experiment_list):
    compilation_results = []

    for experiment_name in experiment_list:
        first_dataset = False
        for protocol in data_experiments[experiment_name]["protocols"]:
            for repetition in range(data_experiments[experiment_name]["n_repetitions"]):
                if not first_dataset or data_experiments[experiment_name]["change_dataset"]:
                    if experiment_name.startswith("real_experiment"):
                        dataset_name = data_experiments[experiment_name]["dataset"]
                        path_train = config["experiments_path"] + "real_experiment" + \
                            "/datasets/" + dataset_name + "_train_" + str(repetition) + ".csv"
                        path_test = config["experiments_path"] + "real_experiment" + \
                            "/datasets/" + dataset_name + "_test_" + str(repetition) + ".csv"
                        first_dataset = True
                    else:
                        dataset_name = "toy_dataset"
                        path_train = config["experiments_path"] + experiment_name + \
                            "/datasets/" + dataset_name + "_train_" + str(repetition) + ".csv"
                        path_test = config["experiments_path"] + experiment_name + \
                            "/datasets/" + dataset_name + "_test_" + str(repetition) + ".csv"
                        first_dataset = True

                for algorithm in data_experiments[experiment_name]["algorithms"]:
                    output_file_name = "ouput_secure_" + \
                        algorithm["name"] + "_" + \
                        protocol["name"] + "_" + str(repetition) + ".txt"
                    output_file_path = config["experiments_path"] + experiment_name + "/" + output_file_name
                    
                    # Reading dataset used in experiment 
                    X_train, y_train = analyzer_utils.load_dataset(path_train) 
                    X_test, y_test = analyzer_utils.load_dataset(path_test)
                    
                    if algorithm["name"] == "logreg":
                        y_train[y_train == -1] = 0
                        y_test[y_test == -1] = 0

                    # Extracting training performance data
                    sec_time, data_sent, global_data_sent = analyzer_utils.get_execution_info(output_file_path)
                    
                    # Creating clear model
                    model_clear = analyzer_utils.get_model_clear_with_params(algorithm)
                    
                    # Testing clear model
                    time_a = datetime.datetime.now()
                    model_clear.fit(X_train, y_train)
                    clean_time_date = datetime.datetime.now() - time_a
                    clean_time = float(str(clean_time_date.seconds) + "." + str(clean_time_date.microseconds))
                    clean_train_score = model_clear.score(X_train, y_train)
                    clean_test_score = model_clear.score(X_test, y_test)
                    
                    # Testing secure model
                    # Creating the clear model
                    print(algorithm)
                    parameters = analyzer_utils.get_parameters(output_file_path)
                    model_secure = analyzer_utils.load_parameters(parameters, algorithm, X_train, y_train)
                    sec_train_score = model_secure.score(X_train, y_train)
                    sec_test_score = model_secure.score(X_test, y_test)

                    # Computing ring size
                    ring_size = compute_ring_size(*compute_precision(experiment_name, algorithm))

                    experiment_results = [
                        # Experiment description
                        experiment_name, 
                        protocol["name"],
                        algorithm["name"],
                        data_experiments[experiment_name]["n_rows"],
                        data_experiments[experiment_name]["n_columns"],
                        data_experiments[experiment_name]["class_sep"],
                        repetition,

                        # Experiment results
                        sec_train_score,
                        sec_test_score,
                        clean_train_score,
                        clean_test_score,
                        sec_time,
                        clean_time,
                        ring_size,
                        data_sent,
                        global_data_sent
                    ]

                    compilation_results.append(experiment_results)

    save_experiment_results(compilation_results)


def save_experiment_results(results):
    headers = [
        # Experiment description
        "Experiment",
        "Protocol",
        "Algorithm",
        "Rows",
        "Columns",
        "Class sep",
        "Repetition",

        # Experiment results
        "Train acc secure",
        "Test acc secure",
        "Train acc clear",
        "Test acc clear",
        "Time secure",
        "Time clean",
        "Ring size",
        "Data sent",
        "Global data sent"
    ]

    df_results = pd.DataFrame(
        data=np.array(results),
        columns=headers
    )
        
    df_results.to_csv(config["experiments_path"] + "reports/results_experiments-" + \
        str(datetime.datetime.now()) + ".csv")



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

    extract_results(experiment_list)
