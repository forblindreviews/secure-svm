from imp import PKG_DIRECTORY
import numpy as np
from fxpmath import Fxp
import math

class FxpDualLSSVMDebug(object):
    
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

        # Initialization of beta_k
        beta_k = Fxp(
            val=np.zeros(shape=(self.data.shape[0] + 1, 1)) + 0.5,
            like=self.FXP_FORMAT
        )

        for i in range(self.max_iter):
            print("################################################################")
            print("======================= [ ITERATION", i," ] =======================")
            p_k = opt_vect - opt_matrix.dot(beta_k)
            print("==> Showing p_k:")
            print(p_k)
            print("-------------------------------------")
            
            r_k = p_k.T.dot(p_k) / p_k.T.dot(opt_matrix.dot(p_k))
            print("==> Showing r_k quotient:")
            print(r_k)
            print("-------------------------------------")

            beta_k = beta_k + (1 - self.lr) * r_k * p_k
            print("==> Showing beta_k new:")
            print(beta_k)
            print("-------------------------------------")

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


if __name__ == "__main__":
    X = np.array([[-1.381895893715888, 0.01981742410253151],
        [2.3793562361570264, -1.0748016042019966],
        [2.9451908614425815, 3.0787780573148913],
        [-0.1409384031152766, 1.585858639514726]])

    y = np.array([[1],
        [-1],
        [-1],
        [1]])

    k = 102
    f = 25

    fxp_svm = FxpDualLSSVMDebug(lambd=10, lr=0.1, max_iter=50, length_word=k, length_frac=f)
    fxp_svm.fit(X, y)