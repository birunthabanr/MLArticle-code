import numpy as np
import matplotlib.pyplot as plt

# Define the cost functions
def cost_positive(p):
    return -np.log(p)

def cost_negative(p):
    return -np.log(1 - p)

# Define probabilities
p = np.linspace(0.01, 0.99, 100)

# Compute costs
cost_pos = cost_positive(p)
cost_neg = cost_negative(p)

# Plot the cost function for y=1 (Positive Instance)
plt.figure(figsize=(10, 6))
plt.plot(p, cost_pos, color='blue', label='Cost for y=1')
plt.xlabel('Predicted Probability')
plt.ylabel('Cost')
plt.title('Cost Function for Positive Instances (y=1)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the cost function for y=0 (Negative Instance)
plt.figure(figsize=(10, 6))
plt.plot(p, cost_neg, color='orange', label='Cost for y=0')
plt.xlabel('Predicted Probability')
plt.ylabel('Cost')
plt.title('Cost Function for Negative Instances (y=0)')
plt.legend()
plt.grid(True)
plt.show()
