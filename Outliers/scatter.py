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
plt.scatter(X, y_with_outliers, color='skyblue', label='Data with Outliers')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression Line')
plt.title('Scatter Plot with Linear Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.show()
