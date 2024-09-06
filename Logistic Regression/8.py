import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = datasets.load_iris()
X = iris["data"][:, 2:4]  # petal length and petal width
y = (iris["target"] == 2).astype(int)  # 1 if Iris virginica, else 0

# Train a Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Generate a grid of values over the feature space
x0, x1 = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500)
)
X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = log_reg.predict_proba(X_new)[:, 1].reshape(x0.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.contour(x0, x1, y_proba, levels=[0.5], colors='black', linestyles='--')

# Plot the original data points
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', marker='o', label="Iris virginica (Actual)")
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='x', label="Not Iris virginica (Actual)")

# Labels and title
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Logistic Regression - Iris virginica Classification with Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
