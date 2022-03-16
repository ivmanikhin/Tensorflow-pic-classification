import tensorflow as tf
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt


# Make synthetic dataset
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=12)

# Visualize data
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
print(circles)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)

print(X.shape)
print(y.shape)

print(X[0])
print(y[0])

plt.show()