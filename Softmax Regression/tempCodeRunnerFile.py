import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load MNIST data
mnist = fetch_openml('mnist_784')

# Split data into features and target labels
X, y = mnist['data'], mnist['target'].astype(np.int64)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Softmax Regression model
softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
softmax_reg.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = softmax_reg.predict(X_train_scaled)
y_test_pred = softmax_reg.predict(X_test_scaled)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Output the results
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Plotting the L2 norm variation of weights
import matplotlib.pyplot as plt

# Compute L2 norms of weight changes
coeffs = softmax_reg.coef_
norms = np.linalg.norm(coeffs, axis=1)

plt.plot(norms)
plt.xlabel('Class Index')
plt.ylabel('L2 Norm of Weight Vectors')
plt.title('L2 Norm Variation of Weights Across Classes')
plt.show()
