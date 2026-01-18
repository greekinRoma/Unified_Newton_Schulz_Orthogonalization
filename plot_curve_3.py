import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. CVPR-style global settings
# ==============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,           # body text
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "figure.figsize": (3.3, 2.5),  # single-column width in inches
    "axes.unicode_minus": False
})

# ==============================
# 2. Extreme points evolution
# ==============================
K = 30
k_vals = np.arange(1, K + 1)
x_star = 1 / np.sqrt(2**(k_vals + 1) + 1)
y_star = x_star * (2**(k_vals + 1) / (2**(k_vals + 1) + 1))**(2**k_vals)

fig, ax = plt.subplots()
ax.plot(k_vals, x_star, marker='o', linestyle='-', markersize=4, label=r"$x_k^*$", color='tab:blue')
ax.plot(k_vals, y_star, marker='s', linestyle='--', markersize=4, label=r"$y_k^*$", color='tab:orange')

ax.set_xlabel(r"$k$")
ax.set_ylabel("Extreme point value")
# ax.set_title("Extreme Point Values")
ax.legend(frameon=False, loc='upper right')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ==============================
# Remove extra whitespace
# ==============================
plt.tight_layout()
fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)

plt.savefig("plot_curve_3.pdf", dpi=300)
plt.show()

# ==============================
# 3. Function curves converging to impulse
# ==============================
x = np.linspace(0, 1, 30000)
Ks = [1, 2, 3, 4, 10, 16]

fig, ax = plt.subplots()
for k in Ks:
    y = x * (1 - x**2)**(2**(k-1))
    y = y / y.max()  # normalize max to 1
    ax.plot(x, y, label=f'$k={k}$')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
# ax.set_title(r'$y^{normalized} = x(1-x^2)^{2^{k-1}}/y^{*}$')
ax.legend(frameon=False, fontsize=7, loc='upper right')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ==============================
# Remove extra whitespace
# ==============================
plt.tight_layout()
fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)

plt.savefig("plot_curve_4.pdf", dpi=300)
plt.show()
