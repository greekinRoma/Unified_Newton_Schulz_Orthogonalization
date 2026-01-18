import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. Dense sampling in (0,1)
# ==============================
x = np.linspace(0, 1, 30000)
Ks = [0, 2, 4, 8, 16, 32]  # k values

# ==============================
# 2. Set Times New Roman and figure style for paper
# ==============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,               # body text size
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "figure.figsize": (3.3, 2.5), # width x height in inches
    "axes.unicode_minus": False
})

# ==============================
# 3. Plot
# ==============================
fig, ax = plt.subplots()

for k in Ks:
    y = x * (1 - x**2)**(2**k)
    y = y / y.max()  # normalize max value to 1
    ax.plot(x, y, label=f'$k={k}$')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title(r'$y = x(1-x^2)^{2^k}/x_{*}$')
ax.legend(frameon=False, fontsize=7, loc='upper right')
ax.grid(False)

plt.tight_layout()
plt.savefig("plot_curve_4.pdf")