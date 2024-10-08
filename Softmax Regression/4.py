import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(np.int8)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.transform(X_test.astype(np.float64))

# Initialize list to store weight variations (L-infinity norm changes)
weights_diff = []
train_accuracies = []
test_accuracies = []
n_iterations = 10  # Number of iterations for tracking

# Initial weights (for comparison at the first iteration)
previous_weights = np.zeros((10, X_train.shape[1]))  # 10 classes, 784 features

# Loop over different numbers of iterations to track weight changes
for i in range(1, n_iterations + 1):
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=i, random_state=42)
    softmax_reg.fit(X_train_scaled, y_train)
    
    # Get the current weights
    current_weights = softmax_reg.coef_
    
    # Calculate L-infinity norm (maximum absolute difference)
    weight_change = np.linalg.norm(current_weights - previous_weights, ord=np.inf)
    weights_diff.append(weight_change)
    
    # Update previous weights for the next iteration
    previous_weights = current_weights.copy()
     # Calculate training and testing accuracy
    train_pred = softmax_reg.predict(X_train_scaled)
    test_pred = softmax_reg.predict(X_test_scaled)
    train_accuracies.append(accuracy_score(y_train, train_pred))
    test_accuracies.append(accuracy_score(y_test, test_pred))

# Plotting the L-infinity norm of weight changes across iterations
# plt.plot(range(1, n_iterations + 1), weights_diff, marker='o')
# plt.title('L-Infinity Norm of Weight Changes Across Iterations')
# plt.xlabel('Iteration')
# plt.ylabel('L-Infinity Norm of Weight Change')
# plt.grid(True)
# plt.show()


# Plotting the L-Infinity norm of weight changes across iterations
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, n_iterations + 1), weights_diff, marker='o')
plt.title('L-Infinity Norm of Weight Changes Across Iterations')
plt.xlabel('Iteration')
plt.ylabel('L-Infinity Norm of Weight Change')
plt.grid(True)

# Plotting training and testing accuracy across iterations
plt.subplot(1, 2, 2)
plt.plot(range(1, n_iterations + 1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, n_iterations + 1), test_accuracies, label='Testing Accuracy', marker='x')
plt.title('Training & Testing Accuracy Across Iterations')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()