import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset
iris = load_iris()
X = iris["data"][:, (2, 3)]  # Petal length and petal width
y = iris["target"]

# Train the Softmax Regression model
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)

# Make predictions for a new flower with 5 cm long and 2 cm wide petals
new_flower = [[5, 2]]
prediction = softmax_reg.predict(new_flower)
probabilities = softmax_reg.predict_proba(new_flower)

print(f"Predicted class: {prediction[0]}")
print(f"Class probabilities: {probabilities}")

# Plot decision boundaries
x0, x1 = np.meshgrid(
    np.linspace(0, 7, 500).reshape(-1, 1), np.linspace(0, 3, 200).reshape(-1, 1)
)
X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.title("Decision Boundaries and Probabilities for Iris Versicolor", fontsize=16)
plt.show()
