import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Create values for x-axis
x = np.linspace(-3, 5, 500)

# F2: Standard normal CDF
F2 = norm.cdf(x, loc=0, scale=1)
# F1: Right-shifted normal (dominates F2)
F1 = norm.cdf(x, loc=1.2, scale=1)

plt.figure(figsize=(9, 5))
plt.plot(x, F2, label='F2 (Mean = 0)', color='#E45756', linewidth=2.5)
plt.plot(x, F1, label='F1 (Mean = 1.2)', color='#4E79A7', linewidth=2.5)

# Fill the area between the CDFs for visual emphasis
plt.fill_between(x, F1, F2, where=F2>F1, color='#76B7B2', alpha=0.6, label='F2 - F1')

# Add some annotation to show "F1 dominates F2"
plt.text(1, 0.7, "F1 is always below F2\n⇒ F1 first-order dominates F2",
         fontsize=13, color='#333333', bbox=dict(boxstyle="round,pad=0.3", fc='#F1F1F1', ec='#4E79A7', lw=2))

plt.title('First-Order Stochastic Dominance (SD1):\nF1 Dominates F2', fontsize=16, weight='bold')
plt.xlabel('x', fontsize=14)
plt.ylabel('CDF Value', fontsize=14)
plt.legend(fontsize=12, loc='lower right')
plt.ylim([-0.05, 1.05])
plt.grid(True, linestyle=':', alpha=0.7)

# # Add a pointer/arrow to the space between curves
# plt.annotate('Area where F2 > F1',
#              xy=(2, F2[np.abs(x-2).argmin()]), xycoords='data',
#              xytext=(2.5, 0.35), textcoords='data',
#              arrowprops=dict(facecolor='#E45756', edgecolor='black', arrowstyle='->', lw=2),
#              fontsize=13, color='#E45756', weight='bold')

plt.tight_layout()
plt.show()
