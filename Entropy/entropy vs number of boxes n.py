import numpy as np
import matplotlib.pyplot as plt

def calculate_entropy(prob_win):
    prob_lose = 1 - prob_win
    entropy = -(prob_win * np.log2(prob_win) + prob_lose * np.log2(prob_lose))
    return entropy

num_boxes_full = np.arange(2, 101)  # All points from 2 to 100
num_boxes_highlight = np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

prob_win_full = 1 / num_boxes_full
prob_win_highlight = 1 / num_boxes_highlight

entropies_full = [calculate_entropy(p) for p in prob_win_full]
entropies_highlight = [calculate_entropy(p) for p in prob_win_highlight]

plt.figure(figsize=(10, 6))
plt.plot(num_boxes_full, entropies_full, linestyle="-", color="b", label="Entropy Curve")
plt.scatter(num_boxes_highlight, entropies_highlight, color="b", label="Data Points", zorder=5)
plt.title("Entropy vs. Number of Boxes", fontsize=14)
plt.xlabel("Number of Boxes (n)", fontsize=12)
plt.ylabel("Entropy (H)", fontsize=12)
plt.grid(alpha=0.5)
plt.legend()
plt.show()
