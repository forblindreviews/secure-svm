import numpy as np

class FlpDualSVMMix(object):

    def __init__(self, C, max_passes=1000, kernel="linear", tolerance=1e-1, degree=None) -> None:
        super().__init__()
        self.max_passes = max_passes
        self.degree = degree
        self.C = C
        self.tolerance = tolerance
        self.kernel_type = kernel

    def kernel(self, a, b):
        if self.kernel_type == "linear":
            return a.T.dot(b)[0]
        if self.kernel_type == "poly":
            return np.power(1 + a.T.dot(b)[0], self.degree)
    
    def predict_distance_vect(self, x):
        prediction = 0
        for i in range(self.data.shape[0]):
            Xi = np.expand_dims(self.data[i], axis=1)
            prediction += self.alphas[i][0] * self.y[i][0] * self.kernel(Xi, x)
        
        prediction -= self.b

        return prediction

    def predict_distance(self, X):
        predictions = np.zeros(shape=(X.shape[0], 1))
        for i in range(X.shape[0]):
            Xi = np.expand_dims(X[i], axis=1)
            predictions[i][0] = self.predict_distance_vect(Xi)

        return predictions

    def predict(self, X):
        distances = self.predict_distance(X)
        return np.sign(distances)

    def fit(self, X, y):
        self.data = X
        self.y = y
        
        self.steps = 0

        self.alphas = np.zeros(shape=(self.data.shape[0], 1))
        self.b = 0

        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(self.data.shape[0]):
                next_i = False

                Xi = np.expand_dims(self.data[i], axis=1)
                Ei = self.predict_distance_vect(Xi) - self.y[i][0]
                yi = self.y[i][0]
                alpha_i = self.alphas[i][0]

                ri = Ei * yi

                if (ri < -self.tolerance and alpha_i < self.C) or (ri > self.tolerance and alpha_i > 0):
                    self.steps += 1

                    for j in range(i + 1, self.data.shape[0]):
                        j = self.get_index_heuristic(i)
                        Xj = np.expand_dims(self.data[j], axis=1)
                        Ej = self.predict_distance_vect(Xj) - self.y[j][0]
                        yj = self.y[j][0]

                        alpha_i_old = self.alphas[i][0]
                        alpha_j_old = self.alphas[j][0]

                        # Computing L and H
                        if yi != yj:
                            L = max(0, alpha_j_old - alpha_i_old)
                            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                        else:
                            L = max(0, alpha_j_old + alpha_i_old - self.C)
                            H = min(self.C, alpha_j_old + alpha_i_old)

                        if L == H:
                            next_i = True
                            break
                        
                        kii = self.kernel(Xi, Xi)
                        kij = self.kernel(Xi, Xj)
                        kjj = self.kernel(Xj, Xj)

                        eta = 2 * kij - kii - kjj

                        if eta >= 0:
                            next_i = True
                            break

                        alpha_j_new = alpha_j_old - yj * (Ei - Ej) / eta

                        # Alpha2 new clipped
                        if alpha_j_new < L:
                            alpha_j_new = L
                        elif alpha_j_new > H:
                            alpha_j_new = H

                        if np.abs(alpha_j_old - alpha_j_new) < 1e-5:
                            next_i = True
                            break
                        
                        s = yi * yj
                        alpha_i_new = alpha_i_old + s * (alpha_j_old - alpha_j_new)

                        # Update threshold
                        b1 = self.b - Ei - yi * (alpha_i_new - alpha_i_old) * kii - yj * (alpha_j_new - alpha_j_old) * kij
                        b2 = self.b - Ej - yi * (alpha_i_new - alpha_i_old) * kij - yj * (alpha_j_new - alpha_j_old) * kjj
                        
                        if 0 < alpha_i_new and alpha_i_new < self.C:
                            self.b = b1
                        elif 0 < alpha_j_new and alpha_j_new < self.C:
                            self.b = b2
                        else:
                            self.b = (b1 + b2) / 2.

                        self.alphas[i][0] = alpha_i_new
                        self.alphas[j][0] = alpha_j_new

                        num_changed += 1
                    
                    if next_i:
                        next_i = False
                        continue
            
            if num_changed == 0:
                passes += 1
            else:
                passes = 0

    def get_index_heuristic(self, i):
        j = np.random.randint(0, self.data.shape[0])
        while i == j:
            j = np.random.randint(0, self.data.shape[0])
        return j

    def score(self, X, y_true):
        predict = self.predict(X)
        n_correct = np.sum(predict == y_true)
        return n_correct / X.shape[0]


    

    