import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Seed for reproducibility
np.random.seed(42)

# Generate data
n_samples = 1000
data = np.random.normal(loc=0, scale=1, size=n_samples)

# Introduce outliers
outliers = np.random.normal(loc=50, scale=5, size=5)
data_with_outliers = np.concatenate([data, outliers])

# Calculate mean
mean_with_outliers = np.mean(data_with_outliers)

# # Plot histogram with fitted normal distribution
# plt.figure(figsize=(10, 6))
# sns.histplot(data_with_outliers, kde=False, color='black', stat='density', bins=30)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mean_with_outliers, np.std(data_with_outliers))
# plt.plot(x, p, 'k', linewidth=2)
# plt.title(f'Histogram with Normal Distribution Fit\nMean with Outliers = {mean_with_outliers:.2f}')
# plt.xlabel('Value')
# plt.ylabel('Density')

# plt.show()

# print(f'Mean with Outliers: {mean_with_outliers:.2f}')


from sklearn.linear_model import LinearRegression

# Generate a simple linear relationship dataset
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = 3 * X.squeeze() + np.random.normal(loc=0, scale=1, size=n_samples)

# Introduce outliers in the y variable
y_with_outliers = y.copy()
y_with_outliers[-5:] += 50  # Adding large values to the last few points

# Fit linear regression model
model = LinearRegression()
model.fit(X, y_with_outliers)
y_pred = model.predict(X)

# Scatter plot of data with outliers and linear regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y_with_outliers, color='black', label='Data with Outliers')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression Line')
plt.title('Scatter Plot with Linear Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.show()
