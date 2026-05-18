#!/usr/bin/env python3
"""Visualise per-rank particle decomposition for MixD and develop side-by-side.

Each rank's particles are coloured distinctly.  Reads coords_rank<N>.txt files
written by sphexa after the first domain.sync().

Usage:
    plot_rank_decomposition.py [--backend cpu|gpu] [--test windshock|kelvin]
                               [--proj xy|xz|yz] [--3d] [--out FILE]
"""

import argparse
import glob
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=here
    ).decode().strip()


def load_run_dir(run_dir: str, subsample: float = 1.0) -> "dict[int, np.ndarray]":
    """Return {rank: ndarray(N,3)} from coords_rank*.txt files in run_dir.

    subsample: fraction of particles to keep per rank (0 < subsample <= 1).
    """
    pattern = os.path.join(run_dir, "coords_rank*.txt")
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(re.search(r"coords_rank(\d+)\.txt", p).group(1)))
    if not files:
        raise FileNotFoundError(f"No coords_rank*.txt files found in: {run_dir}")
    rng = np.random.default_rng(0)
    result = {}
    for f in files:
        rank = int(re.search(r"coords_rank(\d+)\.txt", f).group(1))
        data = np.loadtxt(f)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if subsample < 1.0:
            n_keep = max(1, int(len(data) * subsample))
            idx = rng.choice(len(data), size=n_keep, replace=False)
            data = data[idx]
        result[rank] = data
    return result


def scatter_ranks(ax: "plt.Axes", rank_data: "dict[int, np.ndarray]",
                  col_a: int, col_b: int, xlabel: str, ylabel: str,
                  title: str, cmap) -> None:
    ranks = sorted(rank_data)
    n_ranks = len(ranks)
    for i, rank in enumerate(ranks):
        pts = rank_data[rank]
        color = cmap(i / max(n_ranks - 1, 1))
        ax.scatter(pts[:, col_a], pts[:, col_b], s=0.5, color=color,
                   label=f"rank {rank}", rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

def scatter_ranks_3d(ax, rank_data: "dict[int, np.ndarray]",
                     title: str, cmap) -> None:
    ranks = sorted(rank_data)
    n_ranks = len(ranks)
    all_pts = np.vstack(list(rank_data.values()))
    for i, rank in enumerate(ranks):
        pts = rank_data[rank]
        color = cmap(i / max(n_ranks - 1, 1))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, color=color,
                   label=f"rank {rank}", rasterized=True)
    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    ax.set_box_aspect((
        float(all_pts[:, 0].max() - all_pts[:, 0].min()),
        float(all_pts[:, 1].max() - all_pts[:, 1].min()),
        float(all_pts[:, 2].max() - all_pts[:, 2].min()),
    ))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)


# ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Plot per-rank particle decomposition.")
parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
parser.add_argument("--test", choices=["windshock", "kelvin"], default="windshock")
parser.add_argument("--proj", choices=["xy", "xz", "yz"], default="xy",
                    help="Projection plane (default: xy)")
parser.add_argument("--3d", dest="interactive_3d", action="store_true", default=False,
                    help="Also produce an interactive 3D plot (requires plotly)")
parser.add_argument("--subsample", type=float, default=0.1, metavar="FRAC",
                    help="Fraction of particles to plot per rank (default: 0.1)")
parser.add_argument("--out", default="",
                    help="Save figure to this path instead of displaying it")
args = parser.parse_args()

BACKEND = args.backend
TEST    = args.test
STEPS   = 200
N       = 50
RANKS   = 16 if BACKEND == "cpu" else 1
CORES   = 18 if BACKEND == "cpu" else 1

PROJ_AXES = {"xy": (0, 1, "x", "y"),
             "xz": (0, 2, "x", "z"),
             "yz": (1, 2, "y", "z")}
col_a, col_b, xlabel, ylabel = PROJ_AXES[args.proj]

MIXD_ROOT    = find_repo_root()
DEV_ROOT     = os.path.join(os.path.dirname(MIXD_ROOT), "sphexa-develop")
RUN_DIR_NAME = f"{TEST}_r{RANKS}_c{CORES}_s{STEPS}_n{N}_{BACKEND}"

mixd_dir = os.path.join(MIXD_ROOT, f"build_{BACKEND}", RUN_DIR_NAME)
dev_dir  = os.path.join(DEV_ROOT,  f"build_{BACKEND}", RUN_DIR_NAME)

print(f"MixD run dir : {mixd_dir}")
print(f"Develop dir  : {dev_dir}")

mixd_data = load_run_dir(mixd_dir, args.subsample)
dev_data  = load_run_dir(dev_dir,  args.subsample)

# Load full data for accurate axis ranges (not for plotting)
mixd_full = load_run_dir(mixd_dir)
dev_full  = load_run_dir(dev_dir)

print(f"Plotting {sum(len(v) for v in mixd_data.values())} / "
      f"{sum(len(v) for v in mixd_full.values())} MixD particles "
      f"(subsample={args.subsample})")

n_ranks = max(len(mixd_data), len(dev_data))
cmap = plt.get_cmap("tab10" if n_ranks <= 10 else "tab20")

fig, (ax_mixd, ax_dev) = plt.subplots(1, 2, figsize=(14, 6))

scatter_ranks(ax_mixd, mixd_data, col_a, col_b, xlabel, ylabel,
              f"MixD — {TEST.upper()} / {BACKEND.upper()}", cmap)
scatter_ranks(ax_dev,  dev_data,  col_a, col_b, xlabel, ylabel,
              f"develop — {TEST.upper()} / {BACKEND.upper()}", cmap)

# Shared legend (one entry per rank)
handles, labels = ax_mixd.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=n_ranks,
           markerscale=6, frameon=False)

fig.suptitle(f"Rank decomposition after first domain sync  [{args.proj.upper()} projection]",
             fontsize=13)
fig.tight_layout(rect=[0, 0.06, 1, 1])

out_path = args.out or f"rank_decomposition_{TEST}_{BACKEND}_{args.proj}.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")

# ── 3-D matplotlib figure (rotatable in a GUI window) ────────────────────────
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection

fig3d, (ax3_mixd, ax3_dev) = plt.subplots(
    1, 2, figsize=(14, 7),
    subplot_kw={"projection": "3d"}
)
scatter_ranks_3d(ax3_mixd, mixd_data,
                 f"MixD — {TEST.upper()} / {BACKEND.upper()}", cmap)
scatter_ranks_3d(ax3_dev, dev_data,
                 f"develop — {TEST.upper()} / {BACKEND.upper()}", cmap)
handles3, labels3 = ax3_mixd.get_legend_handles_labels()
fig3d.legend(handles3, labels3, loc="lower center", ncol=n_ranks,
             markerscale=6, frameon=False)
fig3d.suptitle("Rank decomposition — 3D view", fontsize=13)
fig3d.tight_layout(rect=[0, 0.06, 1, 1])
out_3d = out_path.replace(".png", "_3d.png")
plt.savefig(out_3d, dpi=150, bbox_inches="tight")
print(f"Saved: {out_3d}")

# ── Optional interactive Plotly figure ───────────────────────────────────────
if args.interactive_3d:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.colors as pc
    except ImportError:
        print("plotly not installed – skipping interactive 3D plot.", file=sys.stderr)
    else:
        fig_plotly = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            subplot_titles=[
                f"MixD — {TEST.upper()} / {BACKEND.upper()}",
                f"develop — {TEST.upper()} / {BACKEND.upper()}",
            ],
        )
        palette = pc.qualitative.Plotly + pc.qualitative.D3

        def _scene_layout(full_rank_data):
            all_pts = np.vstack(list(full_rank_data.values()))
            rx = float(all_pts[:, 0].max() - all_pts[:, 0].min())
            ry = float(all_pts[:, 1].max() - all_pts[:, 1].min())
            rz = float(all_pts[:, 2].max() - all_pts[:, 2].min())
            ref = max(rx, ry, rz)
            return dict(
                aspectmode="manual",
                aspectratio=dict(x=rx / ref, y=ry / ref, z=rz / ref),
                xaxis=dict(range=[float(all_pts[:, 0].min()), float(all_pts[:, 0].max())]),
                yaxis=dict(range=[float(all_pts[:, 1].min()), float(all_pts[:, 1].max())]),
                zaxis=dict(range=[float(all_pts[:, 2].min()), float(all_pts[:, 2].max())]),
            )

        for col_idx, (label, rank_data) in enumerate(
            [("MixD", mixd_data), ("develop", dev_data)], start=1
        ):
            for i, rank in enumerate(sorted(rank_data)):
                pts = rank_data[rank]
                color = palette[i % len(palette)]
                fig_plotly.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="markers",
                        marker=dict(size=1.5, color=color, opacity=0.7),
                        name=f"rank {rank}",
                        legendgroup=f"rank {rank}",
                        showlegend=(col_idx == 1),
                    ),
                    row=1, col=col_idx,
                )
        fig_plotly.update_layout(
            title=f"Rank decomposition — {TEST.upper()} / {BACKEND.upper()}",
            legend=dict(itemsizing="constant"),
            scene=_scene_layout(mixd_full),
            scene2=_scene_layout(dev_full),
        )
        html_path = out_path.replace(".png", "_3d_interactive.html")
        fig_plotly.write_html(html_path, full_html=True, include_plotlyjs="cdn")
        print(f"Saved interactive 3D plot: {html_path}")
