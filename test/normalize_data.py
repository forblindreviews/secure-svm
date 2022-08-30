from statistics import mean
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv("test/bound_results_final.csv")
    N = 1
    group = data["max_denominator"].groupby(data.index // N).max()
    group.hist(bins=20)
    
    
    mean_add = data["max_denominator"].mean() + 4 * data["max_denominator"].std()
    print("Mean plus 4 std =", mean_add)
    
    plt.show()