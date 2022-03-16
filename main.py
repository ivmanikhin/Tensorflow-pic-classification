import tensorflow as tf
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y):
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    y_pred = model.predict(x_in)
    # if len(y_pred[0]) > 1:
    #     y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    # else:
    y_pred = np.round(y_pred).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.pause(0.1)





# Make synthetic dataset
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=12)

# Visualize data
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
print(circles)
print(X[0])
print(y[0])


# make train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Modelling
tf.random.set_seed(32)
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(6, activation=leaky_relu),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["accuracy"])


x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
x_in = np.c_[xx.ravel(), yy.ravel()]
plt.ion()
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.pause(1)

for epoch in range(50):
    print(f"Epoch {epoch}:")
    model.fit(X_train, y_train, epochs=1)
    plot_decision_boundary(model=model, X=X, y=y)

model.evaluate(X_test, y_test)