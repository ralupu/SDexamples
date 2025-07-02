import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-3, 5, 500)
F2 = norm.cdf(x, loc=0, scale=1)
F1 = 0.7 * norm.cdf(x, loc=-0.5, scale=0.8) + 0.3 * norm.cdf(x, loc=2.5, scale=0.7)

# Crossing points
crossings = np.where(np.diff(np.sign(F1 - F2)))[0]
x_cross = x[crossings]

# Create two vertically stacked subplots (CDFs on top, difference below)
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# --- First subplot: CDFs ---
ax1.plot(x, F2, label='F2', color='#E45756', linewidth=2.5)
ax1.plot(x, F1, label='F1', color='#4E79A7', linewidth=2.5)
ax1.fill_between(x, F1, F2, where=(F1>F2), color='#76B7B2', alpha=0.6, label='A: F1 > F2')
ax1.fill_between(x, F2, F1, where=(F2>F1), color='#F1F1F1', alpha=0.8, label='B: F2 > F1')

for xc in x_cross:
    ax1.axvline(x=xc, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(xc+0.05, 0.55, 'cross', rotation=90, color='grey', fontsize=12, va='bottom')

ax1.text(-1.4, 0.75, 'A', fontsize=16, weight='bold', color='#4E79A7')
ax1.text(2.2, 0.25, 'B', fontsize=16, weight='bold', color='#E45756')

ax1.set_title('No First-Order SD: CDFs Cross (as in Figure 1.2)', fontsize=16, weight='bold')
ax1.set_ylabel('CDF Value', fontsize=14)
ax1.legend(fontsize=12, loc='lower right')
ax1.set_ylim([-0.05, 1.05])
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.annotate("F1 > F2 in region A\nF2 > F1 in region B\n→ CDFs cross, no SD1",
            xy=(x_cross[0]+0.6, 0.85), xycoords='data',
            xytext=(2.7, 0.45), textcoords='data',
            # arrowprops=dict(facecolor='#4E79A7', edgecolor='black', arrowstyle='->', lw=2),
            fontsize=13, color='#333333', bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='#4E79A7', lw=2))

# --- Second subplot: Difference F1(x) - F2(x) ---
diff = F1 - F2
ax2.plot(x, diff, color='#A0CBE8', linewidth=2.5)
ax2.axhline(0, color='grey', linestyle='--', linewidth=1.5)
# Shade area A (where F1 > F2)
ax2.fill_between(x, diff, 0, where=(diff>0), color='#76B7B2', alpha=0.6, label='A: F1 > F2')
# Shade area B (where F2 > F1)
ax2.fill_between(x, diff, 0, where=(diff<0), color='#F1F1F1', alpha=0.8, label='B: F2 > F1')

ax2.set_xlabel('x', fontsize=14)
ax2.set_ylabel('F1(x) - F2(x)', fontsize=14)
ax2.set_ylim([-0.55, 0.55])
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(fontsize=12, loc='lower right')

# Annotate A and B
ax2.text(-1.4, 0.32, 'A', fontsize=16, weight='bold', color='#4E79A7')
ax2.text(2.2, -0.38, 'B', fontsize=16, weight='bold', color='#E45756')

plt.tight_layout()
plt.show()
