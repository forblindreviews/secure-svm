import numpy as np
import datetime

X = np.random.random(size=(500, 500))

time_a = datetime.datetime.now()
for i in range(1000):
    Xi = np.expand_dims(X[i % 100], axis=1)
print("Expand time =", datetime.datetime.now() - time_a)

time_a = datetime.datetime.now()
for i in range(1000):
    Xi = X[i % 100]
print("Non-expand time =", datetime.datetime.now() - time_a)