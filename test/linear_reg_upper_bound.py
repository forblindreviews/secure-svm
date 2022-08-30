from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv(
        "bound_results_final_ls.csv"
    )
    
    # Compute logarithm denominator
    log_denominator = np.log2(data.max_denominator)
    data["log2_denominator"] = log_denominator
    
    # Compute maximum for each row/column
    data_group = pd.DataFrame(
        data.groupby(["log2_observations", "log2_characteristics"])["log2_denominator"].max()
    ).reset_index()
    
    # Compute std for each row/column
    deviations = pd.DataFrame(
        data.groupby(["log2_observations", "log2_characteristics"])["log2_denominator"].std()
    ).reset_index()["log2_denominator"]
    
    # Defining dataset
    X = data_group[["log2_observations", "log2_characteristics"]]
    y = data_group["log2_denominator"]

    # Resonse variable adding the deviations
    response_variable = y + 4 * deviations.max()
    
    # Training model
    model = LinearRegression()
    model.fit(X, response_variable)
    print("Coef =", model.coef_)
    print("Intercept =", model.intercept_)
    print("Score =", model.score(X, response_variable))
    

    