import numpy as np
import matplotlib.pyplot as plt

# Define the values of n (number of boxes)
n_values = np.array([2, 3, 5, 8, 10, 20, 30, 50, 80, 100])

# Initialize a list to store entropy values
entropy_values = []

# Calculate entropy for each n
for n in n_values:
    # Probabilities of winning and losing
    p_win = 1 / n
    p_lose = 1 - p_win
    
    # Calculate entropy for this value of n
    entropy = -(p_win * np.log2(p_win) + p_lose * np.log2(p_lose))
    entropy_values.append(entropy)

# Plotting the graph of entropy vs. number of boxes (n)
plt.figure(figsize=(10, 6))
plt.plot(n_values, entropy_values, marker='o', color='b', linestyle='-')
plt.title("Entropy vs. Number of Boxes (n)")
plt.xlabel("Number of Boxes (n)")
plt.ylabel("Entropy (H(P))")
plt.grid(True)
plt.show()
