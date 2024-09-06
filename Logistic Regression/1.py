import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

# Sample data: [Account Age (in months), Number of Support Tickets]
X = np.array([[12, 2], [24, 4], [36, 1], [48, 7], [60, 3]])
# Labels: 1 if the customer churned, 0 if they did not
y = np.array([0, 1, 0, 1, 0])

# Create and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Define the range for the grid
x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Increase the resolution of the grid for a smoother decision boundary
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict using the logistic regression model
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=ListedColormap(['#FF0000', '#0000FF']), label='Training Data')

# Plot new customer data
X_new = np.array([[30, 3], [50, 5]])
y_proba = log_reg.predict_proba(X_new)
plt.scatter(X_new[:, 0], X_new[:, 1], c='green', marker='x', s=100, label='New Customers')

plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Account Age (in months)')
plt.ylabel('Number of Support Tickets')
plt.legend(loc='best')
plt.show()

# Print predicted probabilities for new customers
print("Probabilities of Churn for new customers:", y_proba[:, 1])
