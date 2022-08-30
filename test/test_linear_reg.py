from sklearn.linear_model import LinearRegression
import numpy as np

if __name__ == "__main__":
    # Dataset
    s = 3
    t = 5
    X_raw = []
    y_raw = []
    for i in range(100):
        rand_a = 5 * np.random.random()
        rand_b = 5 * np.random.random()
        X_raw.append([rand_a, rand_b])
        y_raw.append([rand_a ** s * rand_b ** t])
        
    X = np.array(X_raw)
    y = np.array(y_raw)
    print("X shape =", X.shape)
    print("y shape =", y.shape)        
    
    log_y = np.log2(y)
    log_X = np.log2(X)
    
    lr_model = LinearRegression()
    lr_model.fit(log_X, log_y)
    print("Model =", lr_model.coef_, lr_model.intercept_)
    print("Score =", lr_model.score(log_X, log_y))
    
    