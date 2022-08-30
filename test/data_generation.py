from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_dataset(n_samples, n_features):
    X, y = datasets.make_classification(
        n_samples, 
        n_features, 
        n_redundant=0, 
        n_repeated=0,
        n_informative=2,
        class_sep=1,
    )

    y = pd.Series(y).map({0: -1, 1: 1}).values

    # Extend y columns
    y = np.expand_dims(y, axis=1)

    # Save dataset for MATLAB testing
    # df_save = pd.DataFrame(data=np.append(X, y, axis=1))
    # df_save.to_csv("source/datasets/toy_dataset.csv", index=False, columns=None)

    return X, y


if __name__ == "__main__":
    X, y = generate_dataset(500, 2)

    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=y.reshape(-1), cmap='viridis')
    plt.show()