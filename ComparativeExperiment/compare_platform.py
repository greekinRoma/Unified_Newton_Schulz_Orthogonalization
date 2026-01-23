from Our import EfficientPolyModel, PolyModel
from MuonCurve import MuonCurve
from NS import OriginNSCurve
from Cesista import CesistaCurve
from CANS_curve import curve_iteration

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ==============================
# 0. Global style (CVPR-like)
# ==============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "axes.linewidth": 1.0
})

# ==============================
# CVPR / Okabe–Ito color palette
# ==============================
CVPR_COLORS = {
    "our": "#0072B2",        # blue
    "efficient": "#E69F00",  # orange
    "muon": "#009E73",       # green
    "origin": "#D55E00",     # vermillion
    "cesista": "#CC79A7",    # purple
    "cans": "#7F7F7F",       # gray
}

if __name__ == "__main__":

    # ==============================
    # 1. Models
    # ==============================
    N1 = 14
    N2 = 10

    model_our = PolyModel(N=N1, auto_load=True)
    efficient_model_our = EfficientPolyModel(N=N2, auto_load=True)

    model_muon = MuonCurve()
    cesistacurve = CesistaCurve()
    origin_curve = OriginNSCurve()

    x = torch.linspace(0.0, 1.0, 1000)

    # ==============================
    # 2. Forward
    # ==============================
    with torch.no_grad():
        x_np = x.numpy()
        y_our = model_our(x).cpu().numpy()
        # y_efficient = efficient_model_our(x).cpu().numpy()
        y_muon = model_muon.forward(x).cpu().numpy()
        y_cesista = cesistacurve.forward(x).cpu().numpy()
        y_origin = origin_curve.forward(x).cpu().numpy()
        y_cans = curve_iteration(x_np, n=4, preprocess_iters=1, degree=5)

    # ==============================
    # 3. Main figure (LONG STRIP)
    # ==============================
    fig, ax = plt.subplots(figsize=(8.0, 3.2), constrained_layout=True)

    ax.plot(x_np, y_our,
            linewidth=2.2,
            color=CVPR_COLORS["our"],
            label="Our PolyModel")

    # ax.plot(x_np, y_efficient,
    #         linewidth=2.0,
    #         color=CVPR_COLORS["efficient"],
    #         label="Efficient PolyModel")

    ax.plot(x_np, y_muon,
            linewidth=2.0,
            color=CVPR_COLORS["muon"],
            label="MuonCurve")

    ax.plot(x_np, y_origin,
            linewidth=2.0,
            color=CVPR_COLORS["origin"],
            label="OriginNSCurve")

    ax.plot(x_np, y_cesista,
            linewidth=2.0,
            color=CVPR_COLORS["cesista"],
            label="CesistaCurve")

    ax.plot(x_np, y_cans,
            linewidth=2.0,
            color=CVPR_COLORS["cans"],
            label="CANS Curve")

    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$f(x)$", fontsize=12)

    # Legend: horizontal, above figure
    # ==============================
    # Legend (optimized for CVPR)
    # ==============================
    handles, labels = ax.get_legend_handles_labels()

    label_map = {
        "Our PolyModel": "Ours Matmul:15",
        "Efficient PolyModel": "Ours (Efficient) Matmul:9",
        "MuonCurve": "Muon's NS Matmul:15",
        "OriginNSCurve": "Original NS Matmul:16",
        "CesistaCurve": "Cesista's NS Matmul:15",
        "CANS Curve": "CANS Matmul:15",
    }

    labels = [label_map[l] for l in labels]
    label_to_handle = dict(zip(labels, handles))

    order = [
        "Original NS Matmul:16",
        "Muon's NS Matmul:15",
        "Cesista's NS Matmul:15",
        "CANS Matmul:15",
        "Ours Matmul:15",
    ]

    handles = [label_to_handle[l] for l in order]

    ax.legend(
        handles,
        order,                   # 6 个条目 → 3 列 = 2 行
        fontsize=8.5,
        handlelength=2.2,
        handletextpad=0.6,
    )
    ax.set_xlim(0, 1)
    # Full frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    # ==============================
    # 4. Zoom region (0 – 0.1)
    # ==============================
    x1, x2 = 0.0, 0.1
    mask = (x_np >= x1) & (x_np <= x2)

    y_min = min(
        y_our[mask].min(),
        # y_efficient[mask].min(),
        y_muon[mask].min(),
        y_origin[mask].min(),
        y_cesista[mask].min(),
        y_cans[mask].min(),
    )
    y_max = max(
        y_our[mask].max(),
        # y_efficient[mask].max(),
        y_muon[mask].max(),
        y_origin[mask].max(),
        y_cesista[mask].max(),
        y_cans[mask].max(),
    )

    pad = 0.03 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    # ==============================
    # 5. Inset (horizontal-friendly)
    # ==============================
    axins = ax.inset_axes([0.3, 0.02, 0.45, 0.55])

    axins.plot(x_np, y_our, linewidth=1.4, color=CVPR_COLORS["our"])
    # axins.plot(x_np, y_efficient, linewidth=1.2, color=CVPR_COLORS["efficient"])
    axins.plot(x_np, y_muon, linewidth=1.2, color=CVPR_COLORS["muon"])
    axins.plot(x_np, y_origin, linewidth=1.2, color=CVPR_COLORS["origin"])
    axins.plot(x_np, y_cesista, linewidth=1.2, color=CVPR_COLORS["cesista"])
    axins.plot(x_np, y_cans, linewidth=1.2, color=CVPR_COLORS["cans"])

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
        ec="black",
        linewidth=1.5
    )

    # ==============================
    # 6. Save & show
    # ==============================
    fig.savefig(
        "comparison_long_strip.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.show()
