import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
X = np.linspace(-10, 10, 100)
# Logistic function
y = 1 / (1 + np.exp(-X))

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(X, y, label='Prediction Function')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
# plt.scatter(0, 0.7, color='blue', s=100, label='Class 1 (Male) Example: 0.7')
# plt.scatter(0, 0.2, color='green', s=100, label='Class 2 (Female) Example: 0.2')

# plt.scatter(0, 0.7, color='blue', )
# plt.scatter(0, 0.2, color='green', )
plt.xlabel('Input Feature')
plt.ylabel('Probability')
plt.title('Decision Boundary Example')
plt.legend()
plt.grid(True)
plt.show()
