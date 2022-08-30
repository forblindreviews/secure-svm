import numpy as np
from sympy import beta


opt_matrix = np.array([
[5, 4.13381, 0, 0, 0.1, 0.365309, 3.13381, 3.13121],
[4.13381, 7.4359, 0, 0, 1, 1.10656, 5.22276, 5.22171],
[0, 0, 0.01, 0, 0, 0, 0, 0],
[0, 0, 0, 0.01, 0, 0, 0, 0],
[0.1, 1, 0, 0, 1.01, 1, 1, 1],
[0.365309, 1.10656, 0, 0, 1, 1.08775, 1.08192, 1.08116],
[3.13381, 5.22276, 0, 0, 1, 1.08192, 4.22962, 4.21882],
[3.13121, 5.22171, 0, 0, 1, 1.08116, 4.21882, 4.22802]
])

beta_k = np.array([
    [0.384069],
[0.796223],
[0.525672],
[0.730381],
[0.178307],
[0.0578451],
[0.837874],
[0.385408]
])

opt_vect = np.array([
    [5],
[4.13381],
[0.1],
[0.1],
[0.1],
[0.365309],
[3.13381],
[3.13121]
])


for i in range(50):
    p_k = np.dot(opt_matrix, beta_k) - opt_vect
    r_k = np.dot(p_k.T, p_k) / np.dot(p_k.T, np.dot(opt_matrix, p_k))
    
    beta_k = beta_k - (1 - 0.1) * r_k * p_k
    print("Index ===", i)
    print("r_k =", r_k)
    print("numerador =", np.dot(p_k.T, p_k))
    print("denominador =", np.dot(p_k.T, np.dot(opt_matrix, p_k)))
    print(beta_k)
    print("====")
