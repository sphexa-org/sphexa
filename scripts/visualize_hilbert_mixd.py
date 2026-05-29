#!/usr/bin/env python3
"""Visualize a mixed-dimension Hilbert curve by connecting leaf-cell centers."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize MixD Hilbert traversal for a 3D box by connecting leaf centers in key order."
    )
    parser.add_argument("--lx", type=float, required=True, help="Box size along x (must be > 0)")
    parser.add_argument("--ly", type=float, required=True, help="Box size along y (must be > 0)")
    parser.add_argument("--lz", type=float, required=True, help="Box size along z (must be > 0)")

    parser.add_argument(
        "--module-path",
        type=str,
        default="",
        help="Optional path containing the built cstone_sfc module (e.g. build_cpu/domain/python)",
    )
    parser.add_argument(
        "--key-type",
        type=str,
        choices=("uint64_t", "unsigned"),
        default="uint64_t",
        help="Key type to use for MixD operations (default: uint64_t).",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs=3,
        metavar=("BX", "BY", "BZ"),
        default=None,
        help="Optional explicit (bx, by, bz). If omitted, derived from box dimensions.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1073741825,
        help="Safety cap for plotted points (default: 200000).",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Use every N-th key to downsample the polyline (default: 1).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="If set, save figure to this file instead of showing interactively.",
    )
    return parser.parse_args()


def import_bindings(module_path: str):
    if module_path:
        sys.path.insert(0, str(Path(module_path).resolve()))
    try:
        import cstone_sfc  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Failed to import cstone_sfc bindings. Build with -DCSTONE_BUILD_PYTHON_BINDINGS=ON "
            "and optionally pass --module-path to this script."
        ) from exc
    return cstone_sfc


def validate_lengths(lx: float, ly: float, lz: float) -> None:
    if lx <= 0 or ly <= 0 or lz <= 0:
        raise SystemExit("All box lengths must be strictly positive")


def validate_bits(bits: tuple[int, int, int], max_level: int) -> None:
    bx, by, bz = bits
    if min(bits) < 0 or max(bits) > max_level:
        raise SystemExit(f"Bits must be within [0, {max_level}] but got {bits}")


def main() -> None:
    args = parse_args()
    validate_lengths(args.lx, args.ly, args.lz)
    if args.sample_step < 1:
        raise SystemExit("--sample-step must be >= 1")

    cstone_sfc = import_bindings(args.module_path)

    box_limits = (0.0, args.lx, 0.0, args.ly, 0.0, args.lz)
    print(f"args.key_type={type(args.key_type)}, box_limits={box_limits}")
    max_level = int(cstone_sfc.maxTreeLevel(args.key_type))
    print(f"Using max_level={max_level} for key_type={args.key_type}")

    if args.bits is None:
        bx, by, bz = map(int, cstone_sfc.getBoxMixDimensionBits(box_limits, args.key_type))
        print(f"Derived bits from box dimensions: (bx, by, bz)=({bx}, {by}, {bz}), key_type={args.key_type}")
    else:
        bx, by, bz = args.bits
    validate_bits((bx, by, bz), max_level)

    total_leaves = 1 << (3 * max_level)
    plotted_points = math.ceil(total_leaves / args.sample_step)
    if plotted_points > args.max_points:
        raise SystemExit(
            f"Refusing to allocate {plotted_points} points (total leaves={total_leaves}). "
            "Increase --sample-step or --max-points."
        )
    print(f"Total leaves in full tree: {total_leaves}, plotting every {args.sample_step}-th key for {plotted_points} points.")

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for key in range(0, total_leaves, args.sample_step):
        px, py, pz = map(int, cstone_sfc.decodeHilbertMixD(key, bx, by, bz, args.key_type))
        ibox = (px, px + 1, py, py + 1, pz, pz + 1)
        center, size = cstone_sfc.centerAndSizeMixD(ibox, box_limits, bx, by, bz, args.key_type)
        if size[0] <= 0 and size[1] <= 0 and size[2] <= 0:
            continue
        xs.append(float(center[0]))
        ys.append(float(center[1]))
        zs.append(float(center[2]))
        if len(xs) % 100000 == 0:
            print(f"Processed {len(xs)} points...")

    print(f"First point: ({xs[0]}, {ys[0]}, {zs[0]}), Last point: ({xs[-1]}, {ys[-1]}, {zs[-1]}), Total points: {len(xs)}")

    output_path = Path(args.save) if args.save else None
    ext = output_path.suffix.lower() if output_path else ''

    if output_path and ext in {'.png', '.jpg', '.jpeg'}:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, linewidth=1.0, marker="o", markersize=2)
        # Mark start (green) and end (red)
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color="green", s=60, label="start")
        ax.scatter([xs[-2]], [ys[-2]], [zs[-2]], color="red", s=60, label="end")
        ax.legend()
        ax.set_title(
            f"MixD Hilbert curve centers (bx, by, bz)=({bx}, {by}, {bz}), key_type={args.key_type}, "
            f"sample_step={args.sample_step}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect((args.lx, args.ly, args.lz))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
        print(f"Saved static image to {output_path}")
    else:
        # Plotly interactive HTML export or show
        import plotly.graph_objs as go
        import plotly.io as pio

        # Plotly interactive 3D polyline (not closed)
        trace = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(width=2, color='blue'),
            name='Hilbert Curve'
        )
        # Markers for start and end
        start_marker = go.Scatter3d(
            x=[xs[0]], y=[ys[0]], z=[zs[0]],
            mode='markers',
            marker=dict(size=7, color='green', symbol='circle'),
            name='start'
        )
        end_marker = go.Scatter3d(
            x=[xs[-2]], y=[ys[-2]], z=[zs[-2]],
            mode='markers',
            marker=dict(size=7, color='red', symbol='circle'),
            name='end'
        )
        layout = go.Layout(
            title=f"MixD Hilbert curve centers (bx, by, bz)=({bx}, {by}, {bz}), key_type={args.key_type}, sample_step={args.sample_step}",
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                aspectmode='manual',
                aspectratio=dict(x=args.lx, y=args.ly, z=args.lz)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig = go.Figure(data=[trace, start_marker, end_marker], layout=layout)

        if output_path:
            html_path = output_path.with_suffix('.html')
            html_path.parent.mkdir(parents=True, exist_ok=True)
            pio.write_html(fig, file=str(html_path), auto_open=False, include_plotlyjs='embed')
            print(f"Saved interactive plot to {html_path}")
        else:
            fig.show()


if __name__ == "__main__":
    main()
