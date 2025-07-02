import numpy as np
import matplotlib.pyplot as plt

# F1: everyone gets $10
F1_values = np.full(10, 10)

# F2: 7 get $2, 3 get $50
F2_values = np.array([2]*7 + [50]*3)

# Sort values for CDF plot
F1_sorted = np.sort(F1_values)
F2_sorted = np.sort(F2_values)

# Empirical CDFs
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

x1, y1 = ecdf(F1_sorted)
x2, y2 = ecdf(F2_sorted)

plt.figure(figsize=(8, 5))
plt.step(x1, y1, where='post', color='#4E79A7', linewidth=2.5, label='F1: Everyone gets $10')
plt.step(x2, y2, where='post', color='#E45756', linewidth=2.5, label='F2: 7 get $2, 3 get $50')

# Highlight the regions
plt.fill_betweenx(y2, x2, x1[-1], where=(x2<=10), color='#76B7B2', alpha=0.5, label='Many in F2 below $10')
plt.fill_betweenx(y2, x2, x1[-1], where=(x2>10), color='#F1F1F1', alpha=0.5, label='Some in F2 above $10')

plt.axvline(10, color='grey', linestyle='--', alpha=0.7)
plt.text(10.5, 0.3, '$10\nSafe floor', color='grey', fontsize=13)

plt.xlabel('Money Received ($)', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.title('Empirical CDFs: F1 (Safe, Equal) vs. F2 (Risky, Unequal)', fontsize=15, weight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.ylim([0, 1.05])
plt.xlim([0, 55])
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
