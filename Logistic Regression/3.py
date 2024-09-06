import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Generate synthetic data with a linear decision boundary
np.random.seed(0)
n_samples = 200
X = np.random.randn(n_samples, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Class 1 for above the line, Class 0 for below

# Logistic regression model prediction function
def predict_proba(X):
    # Simulate a logistic regression model with a linear decision boundary
    weights = np.array([1, 1])  # Coefficients for our model
    bias = -0.5  # Intercept
    z = np.dot(X, weights) + bias
    return 1 / (1 + np.exp(-z))

# Create a mesh grid for plotting decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(10, 6))
cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_contour = ListedColormap(['#FF0000', '#0000FF'])

plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=cmap_background, alpha=0.3)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_contour, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary in Logistic Regression (Linear Boundary)')
plt.colorbar(label='Probability')

plt.show()
