import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.abspath(""))
from source.floating_svm.flp_dual_svm_ls_BB88 import FlpDualLSBBSVM
from source.experiments.dataset_generator import generate_dataset

# Table headers will be:
# log_2(observations), log_2(characteristics), log_2(max_denominator), log_2(cota_nuestra)

if __name__ == "__main__":
    
    # Observations and characteristics are powers of two
    # The observations may range between 2**5 and 2**12. On the
    # other hand, characteristics may range between 2**1 and 2**5
    sampling_b = 800
    sampling_X = 30
    bins_number = 50
    N_trials = 2000
    limit_exponent_obs = 5
    limit_exponent_char = 1
    
    table_results = []
    
    for exponent_observations in range(5, limit_exponent_obs + 1):
        observations = 2 ** exponent_observations
        
        for exponent_characteristics in range(1, limit_exponent_char + 1):            
            characteristics = 2 ** exponent_characteristics
            
            print(f"Running for {observations} rows and {characteristics} columns")
            
            for trial in range(N_trials):
                list_norms_numerator = []
                list_norms_denominator = []
                ratio = []
                
                print(f"     Running trial {trial}")
                
                for i in range(sampling_X):
                    # Generating X y y with make_classification
                    X, y = generate_dataset(observations, characteristics, 1.0)
                                        
                    # Building SVM instance
                    svm = FlpDualLSBBSVM(lambd=1 / characteristics)
                    svm.data = X
                    svm.y = y
                    
                    # Building A
                    omega = svm.compute_omega()
                    A = svm.compute_A(omega, y)
                    
                    # Norms containers
                    A_norm = np.linalg.norm(A, ord=2)

                    # Some matrices for computing p_k
                    opt_matrix = np.dot(A.T, A)
                    ones_hat = np.concatenate(
                        (np.array([[0]]), np.ones(shape=(svm.data.shape[0], 1))), 
                        axis=0
                    )
                    opt_vect = np.dot(A.T, ones_hat)

                    # First iteration
                    for iter_idx in range(sampling_b):
                        
                        b_i = np.random.random(size=(svm.data.shape[0] + 1, 1))
                        b_i_minus1 = p_i = 0
                        
                        p_i_minus1 = p_i
                        p_i = np.dot(opt_matrix, b_i) - opt_vect

                        delta_p = p_i - p_i_minus1
                        delta_b = b_i - b_i_minus1
                        
                        numerator = np.dot(delta_p.T, delta_b)[0][0]
                        denominator = np.dot(delta_p.T, delta_p)[0][0]
                        r_i = numerator / denominator
                        
                        list_norms_numerator.append(numerator)
                        list_norms_denominator.append(denominator)
                        ratio.append(r_i)

                vector_order = pd.DataFrame({'Norm_numerator': list_norms_numerator,
                                        'Norm_denominator': list_norms_denominator,
                                        'Ratio_r': ratio
                                        })

                # Computing maximum of norms
                max_numerator = vector_order.Norm_numerator.max()
                max_denominator = vector_order.Norm_denominator.max()
                min_r = vector_order.Ratio_r.min()
                
                data_sample = [
                    exponent_observations, 
                    exponent_characteristics,
                    trial,
                    max_numerator,
                    max_denominator,
                    min_r,
                    (observations ** 5) * (characteristics ** 4)
                ]
                
                table_results.append(data_sample)
                
                with open("test/bound_results_preliminar.csv", "a", newline="") as csv_f:
                    writer = csv.writer(csv_f)
                    writer.writerow(data_sample)
                
            
            # print(f'Maximum norm of numerator: {max_numerator:.2E}')
            # print(f'Maximum norm of denominator: {max_denominator:.2}')
            # print('Minimum ratio r:', min_r)
            # print('Matrix A norm:', A_norm)
    
    df_results = pd.DataFrame(
        data=table_results,
        columns=["log2_observations", "log2_characteristics", "trial", "max_numerator", "max_denominator", "min_r", "bound_estimation"]
    )
    
    df_results.to_csv("test/bound_results_final.csv")    
    
    # Plot histogram
    hist = df_results["max_denominator"].hist(bins=bins_number)
    plt.show()
