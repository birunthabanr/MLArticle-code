import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = datasets.load_iris()
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(int)  # 1 if Iris virginica, else 0

# Train a Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Generate new data for predictions
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

# Calculate the decision boundary (where probability is 0.5)
decision_boundary = X_new[np.argmax(y_proba[:, 1] >= 0.5)]

# Plot the estimated probabilities
plt.figure(figsize=(8, 6))
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris virginica")

# Plot the original data points
plt.scatter(X[y == 1], y[y == 1], c='green', marker='o', label="Iris virginica (Actual)")
plt.scatter(X[y == 0], y[y == 0], c='blue', marker='x', label="Not Iris virginica (Actual)")

# Plot the decision boundary
plt.axvline(x=decision_boundary, color='red', linestyle='--')

# Labels and title
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.title("Logistic Regression - Iris virginica Classification with Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
