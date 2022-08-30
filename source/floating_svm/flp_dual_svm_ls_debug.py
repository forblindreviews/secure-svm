import numpy as np
import datetime
import math 
import pandas as pd

class FlpDualLSSVM(object):

    def __init__(self,
                    lambd=4,
                    lr=0.1,
                    max_iter=50,
                    kernel="linear",
                    degree=None) -> None:
        super().__init__()
        self.lr = lr
        self.degree = degree
        self.lambd = lambd
        self.kernel_type = kernel
        self.max_iter = max_iter

    def kernel(self, a, b):
        if self.kernel_type == "linear":
            return a.T.dot(b)[0][0]
        if self.kernel_type == "poly":
            return np.power(1 + a.T.dot(b)[0][0], self.degree)

    def compute_omega(self):
        omega = np.zeros(shape=(self.data.shape[0], self.data.shape[0]))
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[0]):
                Xi = np.expand_dims(self.data[i], axis=1)
                Xj = np.expand_dims(self.data[j], axis=1)
                omega[i][j] = self.y[i][0] * self.y[j][0] * self.kernel(Xi, Xj)
        return omega
    
    def predict_distance_vect(self, x):
        prediction = 0
        for i in range(self.data.shape[0]):
            Xi = np.expand_dims(self.data[i], axis=1)
            prediction += self.alphas[i][0] * self.y[i][0] * self.kernel(Xi, x)
        
        prediction += self.b

        return prediction

    def predict_distance(self, X):
        predictions = np.zeros(shape=(X.shape[0], 1))
        for i in range(X.shape[0]):
            Xi = np.expand_dims(X[i], axis=1)
            predictions[i][0] = self.predict_distance_vect(Xi)

        return predictions

    def predict(self, X):
        predictions = self.predict_distance(X)
        return np.sign(predictions)

    def compute_A(self, omega, y):
        omega_lamba_id = omega + self.lambd * np.identity(self.data.shape[0])
        
        upper_A = np.concatenate((np.array([[0]]), -y.T), axis=1)
        lower_A = np.concatenate((y, omega_lamba_id), axis=1)

        A = np.concatenate((upper_A, lower_A), axis=0)

        return A

    def fit(self, X, y):
        self.data = X
        self.y = y

        # To extract training information
        # self.info = dict()
        # self.info["accuracy"] = list()
        # self.info["pk_norm"] = list()
        
        self.steps = 0
        
        omega = self.compute_omega()

        A = self.compute_A(omega, y)

        opt_matrix = np.dot(A.T, A)
        ones_hat = np.concatenate((np.array([[0]]), np.ones(shape=(self.data.shape[0], 1))), axis=0)
        opt_vect = np.dot(A.T, ones_hat)

        print("################################################################")
        print("======================= [ PREAMBLE ] =======================")
        print("Showing matrix X:")
        print(X)
        print("-------------------------------------")
        print("Showing array y:")
        print(y)
        print("-------------------------------------")
        print("Showing opt_matrix")
        print(opt_matrix)
        print("-------------------------------------")
        print("Showing opt_vect")
        print(opt_vect)
        print("-------------------------------------")

        beta_k = np.zeros(shape=(self.data.shape[0] + 1, 1)) + 0.5
        for i in range(self.max_iter):
            print("################################################################")
            print("======================= [ ITERATION", i," ] =======================")
            p_k = opt_vect - np.dot(opt_matrix, beta_k)
            print("==> Showing p_k:")
            print(p_k)
            print("-------------------------------------")
            r_k = np.dot(p_k.T, p_k) / np.dot(p_k.T, np.dot(opt_matrix, p_k))
            print("==> Showing r_k quotient:")
            print(r_k)
            print("-------------------------------------")
            
            beta_k = beta_k + (1 - self.lr) * r_k * p_k

            print("==> Showing beta_k new:")
            print(beta_k)
            print("-------------------------------------")
            
            self.alphas = beta_k[1:]
            self.b = beta_k[0][0]

            # To extract training information
            # self.info["accuracy"].append(self.score(self.data, self.y))
            # self.info["pk_norm"].append(np.linalg.norm(p_k))
            
            self.steps += 1
            
        self.alphas = beta_k[1:]
        self.b = beta_k[0][0]

        return self.alphas, self.b

    def score(self, X, y_true):
        predict = self.predict(X)
        n_correct = np.sum(predict == y_true)
        return n_correct / X.shape[0]

    def load_parameters(self, alphas, b, X_train, y_train):
        self.alphas = alphas
        self.b = b
        self.data = X_train
        self.y = y_train


def load_train_dataset_from_file(path):
    df_train = pd.read_csv("source/experiments/real_experiment/datasets/dataset_train_0.csv")
    X_train = df_train.iloc[:, :df_train.shape[1] - 1]
    y_train = df_train.iloc[:, df_train.shape[1] - 1]
    y_train = np.expand_dims(y_train, axis=1)
    return X_train.to_numpy(), y_train


if __name__ == "__main__":
    # X = np.array([[-1.381895893715888, 0.01981742410253151],
    #     [2.3793562361570264, -1.0748016042019966],
    #     [2.9451908614425815, 3.0787780573148913],
    #     [-1.381895893715888, 0.01981742410253151],
    #     [2.3793562361570264, -1.0748016042019966],
    #     [2.9451908614425815, 3.0787780573148913],
    #     [-1.381895893715888, 0.01981742410253151],
    #     [2.3793562361570264, -1.0748016042019966],
    #     [2.9451908614425815, 3.0787780573148913],
    #     [-1.381895893715888, 0.01981742410253151],
    #     [2.3793562361570264, -1.0748016042019966],
    #     [2.9451908614425815, 3.0787780573148913],
    #     [-1.381895893715888, 0.01981742410253151],
    #     [2.3793562361570264, -1.0748016042019966],
    #     [2.9451908614425815, 3.0787780573148913],
    #     [-0.1409384031152766, 1.585858639514726]])

    # y = np.array([[1],
    #     [-1],
    #     [-1],
    #     [1],
    #     [-1],
    #     [-1],
    #     [1],
    #     [-1],
    #     [-1],
    #     [1],
    #     [-1],
    #     [-1],
    #     [1],
    #     [-1],
    #     [-1],
    #     [1]])

    path = "source/experiments/real_experiment/datasets/dataset_test_0.csv"
    X, y = load_train_dataset_from_file(path)

    fxp_svm = FlpDualLSSVM(lambd=10, lr=0.1, max_iter=1000)
    fxp_svm.fit(X, y)