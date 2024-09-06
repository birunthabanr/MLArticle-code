import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (1/m) * (-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h))
    return cost.item()  # Ensure we return a scalar value

# Gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        theta = theta - (learning_rate/m) * (X.T @ (sigmoid(X @ theta) - y))
        cost_history[i] = compute_cost(theta, X, y)
    
    return theta, cost_history

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

# Add intercept term to X
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Initialize parameters
theta = np.zeros((X_b.shape[1], 1))
learning_rate = 0.1
iterations = 1000

# Perform gradient descent
theta_opt, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)

# Plot cost function over iterations
plt.plot(range(iterations), cost_history, 'b-')
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='coolwarm')

# Plot the decision boundary
x_values = [np.min(X[:, 0]), np.max(X[:, 0])]
y_values = -(theta_opt[0] + theta_opt[1] * np.array(x_values)) / theta_opt[2]
plt.plot(x_values, y_values, 'k-')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()
