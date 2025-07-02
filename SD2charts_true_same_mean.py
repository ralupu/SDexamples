import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 500)
# Now both have mean 0, but different spreads
F1 = norm.cdf(x, loc=0, scale=0.75)    # More concentrated (less risky)
F2 = norm.cdf(x, loc=0, scale=1.1)     # Wider (riskier)

# Crossing points
crossings = np.where(np.diff(np.sign(F1 - F2)))[0]
x_cross = x[crossings]

diff = F1 - F2
dx = x[1] - x[0]
cum_integral = np.cumsum(diff) * dx

fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, figsize=(11, 13), sharex=True,
    gridspec_kw={'height_ratios': [2, 1, 1]}
)

# --- First subplot: CDFs ---
ax1.plot(x, F1, label='F1', color='#4E79A7', linewidth=2.5)
ax1.plot(x, F2, label='F2', color='#E45756', linewidth=2.5)
ax1.fill_between(x, F1, F2, where=(F1 > F2), color='#76B7B2', alpha=0.6, label='A: F1 > F2')
ax1.fill_between(x, F2, F1, where=(F2 > F1), color='#F1F1F1', alpha=0.8, label='B: F2 > F1')

for xc in x_cross:
    ax1.axvline(x=xc, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(xc+0.05, 0.55, 'cross', rotation=90, color='grey', fontsize=12, va='bottom')

ax1.text(-2.5, 0.7, 'B', fontsize=16, weight='bold', color='#E45756')
ax1.text(2.2, 0.25, 'A', fontsize=16, weight='bold', color='#4E79A7')

ax1.set_title('F2 SSD F1, Equal Means: CDFs Cross, Tighter vs Wider', fontsize=16, weight='bold')
ax1.set_ylabel('CDF Value', fontsize=14)
ax1.legend(fontsize=12, loc='lower right')
ax1.set_ylim([-0.05, 1.05])
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.annotate("F2 > F1 in region B (left)\nF1 > F2 in region A (right)\n→ CDFs cross, F2 SSD F1 but not FSD",
            xy=(x_cross[-1]+0.4, 0.8), xycoords='data',
            xytext=(1.5, 0.45), textcoords='data',
            fontsize=13, color='#333333', bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='#4E79A7', lw=2))

# --- Second subplot: Difference F1(x) - F2(x) ---
ax2.plot(x, diff, color='#A0CBE8', linewidth=2.5)
ax2.axhline(0, color='grey', linestyle='--', linewidth=1.5)
ax2.fill_between(x, diff, 0, where=(diff > 0), color='#76B7B2', alpha=0.6, label='A: F1 > F2')
ax2.fill_between(x, diff, 0, where=(diff < 0), color='#F1F1F1', alpha=0.8, label='B: F2 > F1')

for xc in x_cross:
    ax2.axvline(x=xc, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)

ax2.text(-2.5, 0.18, 'B', fontsize=16, weight='bold', color='#E45756')
ax2.text(2.2, 0.18, 'A', fontsize=16, weight='bold', color='#4E79A7')

ax2.set_ylabel('F1(x) - F2(x)', fontsize=14)
ax2.set_ylim([-0.23, 0.23])
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(fontsize=12, loc='lower right')

# --- Third subplot: Cumulative integral (F1-F2) ---
ax3.plot(x, cum_integral, color='#4E79A7', linewidth=2.5, label=r'$\int_{-\infty}^x [F_1(t) - F_2(t)] dt$')
ax3.axhline(0, color='grey', linestyle='--', linewidth=1.5)

if len(x_cross) >= 1:
    ax3.fill_between(x, cum_integral, 0, where=(x < x_cross[0]), color='#F1F1F1', alpha=0.8, label='B: F2 > F1')
    ax3.fill_between(x, cum_integral, 0, where=(x >= x_cross[0]), color='#76B7B2', alpha=0.6, label='A: F1 > F2')

for xc in x_cross:
    ax3.axvline(x=xc, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.text(xc+0.07, cum_integral[np.abs(x-xc).argmin()]+0.02, 'cross', rotation=90, color='grey', fontsize=12, va='bottom')

ax3.text(-2.5, 0.03, 'B', fontsize=16, weight='bold', color='#E45756')
ax3.text(2.2, 0.02, 'A', fontsize=16, weight='bold', color='#4E79A7')

ax3.set_xlabel('x', fontsize=14)
ax3.set_ylabel(r'$\int_{-\infty}^x [F_1(t)-F_2(t)]dt$', fontsize=15)
ax3.grid(True, linestyle=':', alpha=0.7)
ax3.legend(fontsize=12, loc='lower left')

plt.tight_layout()
plt.show()
