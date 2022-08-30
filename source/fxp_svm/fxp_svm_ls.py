import numpy as np
from fxpmath import Fxp


class FxpDualLSSVM(object):
    
    def __init__(self,
                lambd=4,
                lr=0.1,
                max_iter=50,
                kernel="linear",
                degree=None,
                length_word=16,
                length_frac=8) -> None:
        super().__init__()
        
        self.degree = degree
        self.kernel_type = kernel
        self.max_iter = max_iter
        self.length_word = length_word
        self.length_frac = length_frac

        self.FXP_FORMAT = Fxp(
            val=None,
            signed=True,
            n_word=self.length_word,
            n_frac=self.length_frac,
            rounding="trunc",
            # overflow='wrap'
        )
        self.FXP_FORMAT.config.op_sizing = "same"
        self.FXP_FORMAT.config.const_op_sizing = "same"

        self.lr = Fxp(
            val=lr,
            like=self.FXP_FORMAT
        )
        self.lambd = Fxp(
            val=lambd,
            like=self.FXP_FORMAT
        )

    def kernel(self, a, b):
        if self.kernel_type == "linear":
            return a.T.dot(b)[0][0]

    def compute_omega(self):
        omega = Fxp(
            val=np.zeros(shape=(self.data.shape[0], self.data.shape[0])),
            like=self.FXP_FORMAT
        )

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[0]):
                Xi = Fxp(
                    val=np.expand_dims(self.data[i](), axis=1),
                    like=self.FXP_FORMAT
                )
                Xj = Fxp(
                    val= np.expand_dims(self.data[j](), axis=1),
                    like=self.FXP_FORMAT
                )
                
                omega[i][j] = self.y[i][0] * self.y[j][0] * self.kernel(Xi, Xj)

        return omega

    def compute_A(self, omega):
        identity = Fxp(
            val= np.identity(self.data.shape[0]),
            like=self.FXP_FORMAT
        )

        omega_lamba_id = omega + self.lambd * identity

        upper_A = np.concatenate((np.array([[0]]), -self.y.T()), axis=1)
        lower_A = np.concatenate((self.y(), omega_lamba_id()), axis=1)

        A = Fxp(
            val=np.concatenate((upper_A, lower_A), axis=0),
            like=self.FXP_FORMAT
        )

        return A

    def fit(self, X, y):
        self.data = Fxp(
            val=X,
            like=self.FXP_FORMAT
        )

        self.y = Fxp(
            val=y,
            like=self.FXP_FORMAT
        )

        self.steps = 0

        # Computation of matrix Omega and A
        omega = self.compute_omega()
        A = self.compute_A(omega)

        # Computation of the vectors involved in the optimization
        opt_matrix = A.T.dot(A)
        ones_hat = Fxp(
            val=np.concatenate((np.array([[0]]), np.ones(shape=(self.data.shape[0], 1))), axis=0),
            like=self.FXP_FORMAT
        )
        opt_vect = A.T.dot(ones_hat)

        # Initialization of beta_k
        beta_k = Fxp(
            val=np.random.random(size=(self.data.shape[0] + 1, 1)),
            like=self.FXP_FORMAT
        )

        for i in range(self.max_iter):
            p_k = opt_vect - opt_matrix.dot(beta_k)

            r_k = p_k.T.dot(p_k) / p_k.T.dot(opt_matrix.dot(p_k))

            beta_k = beta_k + (1 - self.lr) * r_k * p_k

            self.alphas = beta_k[1:]
            self.b = beta_k[0][0]

            self.steps += 1

        self.alphas = beta_k[1:]
        self.b = beta_k[0][0]

        return self.alphas, self.b

    def predict_distance_vect(self, x):
        prediction = 0
        for i in range(self.data.shape[0]):
            Xi = np.expand_dims(self.data[i](), axis=1)
            prediction += self.alphas[i][0]() * self.y[i][0]() * self.kernel(Xi, x)
        
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

    def score(self, X, y_true):
        predict = self.predict(X)
        n_correct = np.sum(predict == y_true)
        return n_correct / X.shape[0]
