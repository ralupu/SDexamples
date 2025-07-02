import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ecdf_on_grid(data, grid):
    return np.searchsorted(np.sort(data), grid, side='right') / len(data)

def ssd_statistic(x, y, ngrid=100):
    grid = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), ngrid)
    cdf_x = ecdf_on_grid(x, grid)
    cdf_y = ecdf_on_grid(y, grid)
    dx = grid[1] - grid[0]
    area = np.cumsum(cdf_x - cdf_y) * dx
    stat = np.max(area)
    return stat, area, grid, cdf_x, cdf_y

def bootstrap_ssd(x, y, ngrid=100, n_bootstrap=1000, random_state=42):
    np.random.seed(random_state)
    n_x, n_y = len(x), len(y)
    pooled = np.concatenate([x, y])
    observed_stat, area, grid, cdf_x, cdf_y = ssd_statistic(x, y, ngrid)
    boot_stats = []
    for _ in range(n_bootstrap):
        perm = np.random.choice(pooled, size=n_x + n_y, replace=True)
        boot_x = perm[:n_x]
        boot_y = perm[n_x:]
        stat, _, _, _, _ = ssd_statistic(boot_x, boot_y, ngrid)
        boot_stats.append(stat)
    boot_stats = np.array(boot_stats)
    pval = np.mean(boot_stats >= observed_stat)
    return observed_stat, pval, area, grid, cdf_x, cdf_y

data = pd.read_stata("bitcoin_sp500_daily_rr.dta")
sp500 = data["SP500_daily_rr"].dropna().values
btc = data["BTC_daily_rr"].dropna().values

observed_stat, pval, area, grid, cdf_sp, cdf_btc = bootstrap_ssd(
    sp500, btc, ngrid=100, n_bootstrap=10000)

print(f"SSD test statistic: {observed_stat:.4g}")
print(f"p-value: {pval:.4g}")

# --- Find crossing points
diff = cdf_sp - cdf_btc
crossings = np.where(np.diff(np.sign(diff)))[0]
if len(crossings) > 0:
    cross_idx = crossings[0]
    x_cross = grid[cross_idx]
else:
    cross_idx = None
    x_cross = None

fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, figsize=(10, 12), sharex=True,
    gridspec_kw={'height_ratios': [2, 1, 1]}
)

# 1. Empirical CDFs
ax1.plot(grid, cdf_sp, color='#4E79A7', linewidth=2.5, label='SP500 (F1)')
ax1.plot(grid, cdf_btc, color='#E45756', linewidth=2.5, label='Bitcoin (F2)')
# Shade A (right of cross): F1 > F2
if cross_idx is not None:
    ax1.fill_between(grid, cdf_sp, cdf_btc, where=(diff > 0), color='#76B7B2', alpha=0.6, label='A: SP500 > BTC')
    ax1.fill_between(grid, cdf_btc, cdf_sp, where=(diff < 0), color='#F1F1F1', alpha=0.8, label='B: BTC > SP500')
    ax1.axvline(x_cross, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(x_cross + 0.01, 0.5, 'cross', rotation=90, color='grey', fontsize=12, va='bottom')
else:
    ax1.fill_between(grid, cdf_sp, cdf_btc, where=(diff > 0), color='#76B7B2', alpha=0.6)
    ax1.fill_between(grid, cdf_btc, cdf_sp, where=(diff < 0), color='#F1F1F1', alpha=0.8)

ax1.text(grid[int(0.15*len(grid))], 0.15, 'B', fontsize=16, weight='bold', color='#E45756')
ax1.text(grid[int(0.75*len(grid))], 0.75, 'A', fontsize=16, weight='bold', color='#4E79A7')
ax1.set_title('Empirical CDFs (Grid-based): SP500 (F1) vs Bitcoin (F2)', fontsize=15, weight='bold')
ax1.set_ylabel('Cumulative Probability', fontsize=14)
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, linestyle=':', alpha=0.6)

# 2. Difference of CDFs
ax2.plot(grid, diff, color='#A0CBE8', linewidth=2.5)
ax2.axhline(0, color='grey', linestyle='--', linewidth=1.2)
if cross_idx is not None:
    ax2.fill_between(grid, diff, 0, where=(diff > 0), color='#76B7B2', alpha=0.6)
    ax2.fill_between(grid, diff, 0, where=(diff < 0), color='#F1F1F1', alpha=0.8)
    ax2.axvline(x_cross, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.text(grid[int(0.15*len(grid))], 0.02, 'B', fontsize=16, weight='bold', color='#E45756')
ax2.text(grid[int(0.75*len(grid))], 0.06, 'A', fontsize=16, weight='bold', color='#4E79A7')
ax2.set_ylabel('SP500(x) - Bitcoin(x)', fontsize=14)
ax2.grid(True, linestyle=':', alpha=0.6)

# 3. Cumulative integral of the difference (SSD area)
ax3.plot(grid, area, color='#4E79A7', linewidth=2.5, label=r'Cumulative $\int (F_1-F_2)\,dx$')
ax3.axhline(0, color='grey', linestyle='--', linewidth=1.2)
if cross_idx is not None:
    ax3.fill_between(grid, area, 0, where=(grid < x_cross), color='#F1F1F1', alpha=0.8)
    ax3.fill_between(grid, area, 0, where=(grid >= x_cross), color='#76B7B2', alpha=0.6)
    ax3.axvline(x_cross, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.text(grid[int(0.15*len(grid))], area[int(0.15*len(grid))]*0.8, 'B', fontsize=16, weight='bold', color='#E45756')
ax3.text(grid[int(0.75*len(grid))], area[int(0.75*len(grid))]*0.8, 'A', fontsize=16, weight='bold', color='#4E79A7')
ax3.set_ylabel('Cumulative Integral', fontsize=14)
ax3.set_xlabel('Daily Return', fontsize=14)
ax3.legend(fontsize=11, loc='lower right')
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
