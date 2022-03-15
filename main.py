import sklearn.model_selection
import tensorflow as tf
import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt


# Make synthetic dataset
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=12)

# Visualize data
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
print(circles)
print(X[0])
print(y[0])

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# make train and test datasets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# Modelling
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation=leaky_relu),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=100)

model.evaluate(X_test, y_test)