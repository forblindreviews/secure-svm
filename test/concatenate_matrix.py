import numpy as np

C = 2

omega = np.ones(shape=(4, 4))
y = 100 * np.ones(shape=(4, 1))

omega_lamba_id = omega + C * np.identity(4)
        
upper_A = np.concatenate((np.array([[0]]), y.T), axis=1)
lower_A = np.concatenate((y, omega_lamba_id), axis=1)

A = np.concatenate((upper_A, lower_A), axis=0)

print(A)

ones = np.ones(shape=(4, 1))
print(np.concatenate(
    (np.array([[0]]), np.ones(shape=(4, 1))),
    axis=0)
)