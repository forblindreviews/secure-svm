import numpy as np
from sklearn.metrics import log_loss


class LogisticRegression(object):

    def __init__(self, lr=0.001, threshold=0.5, max_iter=50) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.threshold = threshold


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def transform_data(self, X):
        extra_column = np.ones(shape=(X.shape[0], 1))
        return np.append(X, extra_column, axis=1)


    def compute_grads(self, X, y):
        sig_eval = self.sigmoid(X.dot(self.w))
        grads = X.T.dot(sig_eval - y)
        return grads
    
    
    def compute_error(self, X, y):
        sig_eval = self.sigmoid(X.dot(self.w))
        loss = log_loss(y, sig_eval)
        return loss
    

    def fit(self, X, y):
        # To extract training information
        self.info = {
            "errors": []
        }
        
        X_trans = self.transform_data(X)
        self.w = np.random.rand(X_trans.shape[1], 1)
        
        for i in range(self.max_iter):
            grads = self.compute_grads(X_trans, y)
            self.w = self.w - self.lr * grads
            
            # To extract training information
            self.info["errors"].append(self.compute_error(X_trans, y))
            
        return self.w
    

    def score(self, X, y_true):
        predictions = self.predict(X)
        predictions[predictions >= self.threshold] = 1
        predictions[predictions < self.threshold] = 0
        n_correct = np.sum(predictions == y_true)
        return n_correct / X.shape[0]

    
    def predict(self, X):
        X_trans = self.transform_data(X)
        return self.sigmoid(X_trans.dot(self.w))


    def load_parameters(self, w, threshold):
        self.w = w
        self.threshold = threshold