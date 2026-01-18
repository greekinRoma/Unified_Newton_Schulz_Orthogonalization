import numpy as np
import matplotlib.pyplot as plt
import math
# ==============================
# 0. Font settings (CVPR style)
# ==============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})

# ==============================
# 1. Dense sampling in [0, 1]
# ==============================
num_points = 3000
x = np.linspace(0, 1, num_points, endpoint=True)

# ==============================
# 2. Function list
# ==============================
functions = [
    (lambda x: x, 0, r"$y=x$"),
    (lambda x: x * (1 - x**2), 1, r"$y=x(1-x^2)$"),
    (lambda x: x * (1 - x**2)**2, 2, r"$y=x(1-x^2)^2$"),
    (lambda x: x * (1 - x**2)**3, 3, r"$y=x(1-x^2)^3$"),
    (lambda x: x * (1 - x**2)**4, 4, r"$y=x(1-x^2)^4$"),
    (lambda x: x * (1 - x**2)**5, 5, r"$y=x(1-x^2)^5$"),
]

# ==============================
# 3. Create CVPR-style figure
# ==============================
plt.figure(figsize=(7.5, 3.2))

for func, k, label in functions:
    y = func(x)

    # Plot curve and capture color
    line, = plt.plot(x, y, linewidth=2.2, label=label)
    color = line.get_color()

    # ---- Mark extrema (k >= 1) ----
    if k >= 1:
        x_star = 1 / math.sqrt(2*k + 1)
        y_star = x_star * (1 - x_star**2)**k

        plt.scatter(
            x_star, y_star,
            s=36,
            color=color,
            zorder=5
        )

        plt.annotate(
            rf"$\left(\frac{{1}}{{\sqrt{{{2*k+1}}}}},\,{y_star:.3f}\right)$",
            xy=(x_star, y_star),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=color
        )

# ==============================
# 4. Axis & style
# ==============================
plt.xlim(0, 1)
plt.xlabel(r"$\mathrm{Input}\; x$", fontsize=11)
plt.ylabel(r"$\mathrm{Output}$", fontsize=11)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.legend(
    fontsize=9,
    frameon=False,
    loc="upper right"
)

plt.grid(False)
plt.tight_layout()

# ==============================
# 5. Save & show
# ==============================
plt.savefig("plot_curve_1.pdf")
plt.show()
