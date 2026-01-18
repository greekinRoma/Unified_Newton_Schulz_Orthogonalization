from turtle import color
import torch
import numpy as np
import matplotlib.pyplot as plt

from ComparativeExperiment.Our import PolyModel, EfficientPolyModel
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ==============================
# 0. Global style (CVPR-like)
# ==============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "axes.linewidth": 1.0,
})

# ==============================
# CVPR / Okabe–Ito color palette
# ==============================
CVPR_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#8C564B",  # brown
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#7F7F7F",  # gray
]

# ==============================
# 1. Settings
# ==============================
LIST = [1, 5, 9, 10, 12, 13, 14, 15, 16]
x = torch.linspace(0.0, 1.0, 1000)

# ==============================
# 2. Load models
# ==============================
models_ours = [PolyModel(N=N, auto_load=True) for N in LIST]
models_efficient = [EfficientPolyModel(N=N, auto_load=True) for N in LIST]

# ==============================
# 3. Forward
# ==============================
with torch.no_grad():
    x_np = x.numpy()
    y_ours = [model(x).cpu().numpy() for model in models_ours]
    y_eff  = [model(x).cpu().numpy() for model in models_efficient]

# ==============================
# 4. Plot function (CVPR color optimized)
# ==============================
def plot_with_final_polish(x, ys, Ns, title):

    fig, ax = plt.subplots(
        figsize=(8.0, 3.2),
        constrained_layout=True
    )
    # ---- main curves ----
    for y, N, c in zip(ys, Ns, CVPR_COLORS):
        ax.plot(
            x, y,
            linewidth=2.0,
            color=c,
            label=rf"$N={N}$"
        )

    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$f(x)$", fontsize=12)

    ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=False,
        handlelength=2.5
    )

    # ---- full frame ----
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    # ==============================
    # Zoom region
    # ==============================
    x1, x2 = 0.0, 0.05
    mask = (x >= x1) & (x <= x2)

    y_min = 0.8
    y_max = max(y[mask].max() for y in ys)
    y_max += 0.02 * (y_max - y_min)

    # ==============================
    # Inset
    # ==============================
    axins = ax.inset_axes([0.30, 0.04, 0.5, 0.5])

    for y, c in zip(ys, CVPR_COLORS):
        axins.plot(x, y, linewidth=1.2, color=c)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y_min, y_max)

    axins.set_xticks([])
    axins.set_yticks([])

    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    mark_inset(
        ax, axins,
        loc1=1, loc2=3,
        fc="none",
        color="black",
        linewidth=1.5
    )

    return fig

# ==============================
# 5. Draw & Save
# ==============================
fig1 = plot_with_final_polish(x_np, y_ours, LIST, "PolyModel")
fig2 = plot_with_final_polish(x_np, y_eff,  LIST, "EfficientPolyModel")

plt.show()

fig1.savefig("Curve_5_1.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
fig2.savefig("Curve_5_2.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
