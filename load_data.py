import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="liac-arff",
)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()