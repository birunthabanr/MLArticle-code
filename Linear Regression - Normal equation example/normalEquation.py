import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Generate a simple linear regression problem
X, y = make_regression(
    n_samples=100,      # Number of samples
    n_features=1,       # Only one feature for simplicity
    n_informative=1,    # Only one informative feature
    noise=5,            # A small amount of noise
    random_state=0      # Seed for reproducibility
    )

# Visualize the feature vs target
plt.subplots(figsize=(8, 5))
plt.scatter(X, y, marker='o')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Feature vs Target in a Simple Linear Regression Problem")
plt.show()


# Add x0 = 1 to each instance (for intercept term)
X_b = np.concatenate([np.ones((len(X), 1)), X], axis=1)

# Calculate the normal equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Extract the intercept and coefficients
intercept, coef = theta_best[0], theta_best[1]
print(f"Intercept: {intercept}\nCoefficients: {coef}")

# Making a new sample
new_sample = np.array([[2.0]])

# Adding a bias term to the instance
new_sample_b = np.concatenate([np.ones((len(new_sample), 1)), new_sample], axis=1)

# Predicting the value of our new sample
new_sample_pred = new_sample_b.dot(theta_best)
print(f"Prediction: {new_sample_pred[0]}")



# Validate using Scikit-learn's LinearRegression
lr = LinearRegression()
lr.fit(X, y)

print(f"Intercept (Scikit-learn): {lr.intercept_}")
print(f"Coefficients (Scikit-learn): {lr.coef_}")
print(f"Prediction (Scikit-learn): {lr.predict(new_sample)[0]}")