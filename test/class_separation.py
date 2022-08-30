from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(n_samples, n_features, class_sep=1.0):
    X, y = datasets.make_classification(n_samples, n_features, n_redundant=0, n_informative=2, class_sep=class_sep)
    y = pd.Series(y).map({0: -1, 1: 1}).values

    # Extend y columns
    y = np.expand_dims(y, axis=1)

    return X, y


X, y = generate_dataset(n_samples=1000, n_features=2, class_sep=0.92)
y = np.expand_dims(y, axis=1)

fig, axs = plt.subplots(1, 1)
axs.scatter(X[:,0], X[:,1], c=y, cmap='viridis', alpha=0.7)
axs.set_title("Real dataset")
plt.show()