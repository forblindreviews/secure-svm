from inspect import CO_ASYNC_GENERATOR
import numpy as np
import datetime

class FlpDualGSSVM(object):

    def __init__(self, lambd, max_iter=5000, kernel="linear", tolerance=1e-10, degree=None) -> None:
        super().__init__()
        self.degree = degree
        self.lambd = lambd
        self.tolerance = tolerance
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
                omega[i][j] = self.kernel(Xi, Xj)
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
        
        upper_A = np.concatenate((np.array([[0]]), y.T), axis=1)
        lower_A = np.concatenate((y, omega_lamba_id), axis=1)

        A = np.concatenate((upper_A, lower_A), axis=0)

        return A

    def fit(self, X, y):
        self.data = X
        self.y = y
        
        self.steps = 0
        
        omega = self.compute_omega()

        A = self.compute_A(omega, y)

        opt_matrix = np.dot(A.T, A)
        ones_hat = np.concatenate((np.array([[0]]), np.ones(shape=(self.data.shape[0], 1))), axis=0)
        opt_vect = np.dot(A.T, ones_hat)

        x = np.random.random(size=(self.data.shape[0] + 1, 1))
        for k in range(self.max_iter):
            next_x = np.zeros(x.shape)
            for i in range(next_x.shape[0]):
                accumulator_left = 0
                for j in range(i):
                    accumulator_left += opt_matrix[i][j] * next_x[j][0]
                
                accumulator_right = 0
                for j in range(i + 1, next_x.shape[0]):
                    accumulator_right += opt_matrix[i][j] * x[j][0]
                
                next_x[i][0] = (1 / opt_matrix[i][i]) * (- accumulator_left - accumulator_right + opt_vect[i][0])

            if np.linalg.norm(next_x - x, ord=np.inf) < self.tolerance:
                self.alphas = x[1:]
                self.b = x[0][0]
                return self.alphas, self.b

            x = next_x
        
        self.alphas = x[1:]
        self.b = x[0][0]

        return self.alphas, self.b

    def score(self, X, y_true):
        predict = self.predict(X)
        n_correct = np.sum(predict == y_true)
        return n_correct / X.shape[0]


    

    