import numpy as np
import pandas as pd
import csv

import os
import sys
sys.path.insert(0, os.path.abspath(""))
from source.floating_svm.flp_dual_svm_ls import FlpDualLSSVM
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
    N_trials = 30
    limit_exponent_obs = 10
    limit_exponent_char = 4
    
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
                    
                    # Generating X and y at random
                    # X = 2 * np.random.rand(observations, characteristics) - 1
                    # y = np.random.choice([-1, 1], size = observations, replace = True)
                    # y = np.array(y).reshape(observations, 1)
                    
                    # Building SVM instance
                    svm = FlpDualLSSVM(lambd=1 / characteristics)
                    svm.data = X
                    svm.y = y
                    
                    # Building A
                    omega = svm.compute_omega()
                    A = svm.compute_A(omega, y)
                    
                    # Norms containers
                    A_norm = np.linalg.norm(A, ord=2)

                    # Some matrices for computing p_k
                    opt_matrix = np.dot(A.T, A)
                    ones_hat = np.concatenate((np.array([[0]]), np.ones(shape=(X.shape[0], 1))), axis=0)
                    opt_vect = np.dot(A.T, ones_hat)

                    # First iteration
                    for iter_idx in range(sampling_b):
                        beta_k = np.random.random(size=(X.shape[0] + 1, 1))
                        p_k = np.dot(opt_matrix, beta_k) - opt_vect
                        
                        norm_numerator = np.linalg.norm(p_k, ord=2) ** 2
                        norm_denominator = np.linalg.norm(A.dot(p_k), ord=2) ** 2
                        
                        r_k = norm_numerator / norm_denominator
                        
                        list_norms_numerator.append(norm_numerator)
                        list_norms_denominator.append(norm_denominator)
                        ratio.append(r_k)

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
                    (observations ** 7) * (characteristics ** 6)
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
    # hist = vector_order.hist(bins=bins_number)
    # plt.show()
