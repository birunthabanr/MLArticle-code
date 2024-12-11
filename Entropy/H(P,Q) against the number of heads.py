import numpy as np
import matplotlib.pyplot as plt

# Function to calculate cross-entropy
def cross_entropy(P, Q):
    return -np.sum(P * np.log(Q))

# Original probability distribution (unbiased coin)
P = np.array([0.5, 0.5])

# Store cross-entropy values
cross_entropy_values = []

# Vary the number of heads in 10 coin flips
for heads in range(10):  # Number of heads from 0 to 10
    tails = 10 - heads  # Number of tails
    Q = np.array([heads/10, tails/10])  # Update probability distribution for Q
    H_PQ = cross_entropy(P, Q)  # Calculate cross-entropy for this distribution
    cross_entropy_values.append(H_PQ)

# Plotting the graph
plt.plot(range(10), cross_entropy_values, marker='o', color='b')
plt.title('Cross-Entropy H(P, Q) vs Number of Heads')
plt.xlabel('Number of Heads')
plt.ylabel('Cross-Entropy H(P, Q)')
plt.grid(True)
plt.show()
