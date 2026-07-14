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
        "--level",
        type=int,
        default=0,
        help="Oct number level from right for hilbertMixDIBox (0 = leaves, default: 0).",
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


def count_mixd_nodes_at_level(bx: int, by: int, bz: int, level_from_right: int) -> int:
    """Count MixD boxes at a given level-from-right.

    Piecewise branching in ordered bits b0 <= b1 <= b2:
    - level <= b0: 8-way per level
    - b0 < level <= b1: 4-way per level
    - b1 < level <= b2: 2-way per level
    - level > b2: 1 node
    """
    b0, b1, b2 = sorted((bx, by, bz))
    l = level_from_right

    if l <= b0:
        exponent = bx + by + bz - 3 * l
    elif l <= b1:
        exponent = b1 + b2 - 2 * l
    elif l <= b2:
        exponent = b2 - l
    else:
        exponent = 0

    return 1 << exponent


def main() -> None:
    args = parse_args()
    validate_lengths(args.lx, args.ly, args.lz)
    if args.level < 0:
        raise SystemExit("--level must be >= 0")

    cstone_sfc = import_bindings(args.module_path)

    box_limits = (0.0, args.lx, 0.0, args.ly, 0.0, args.lz)
    print(f"args.key_type={type(args.key_type)}, box_limits={box_limits}")
    max_level = int(cstone_sfc.maxTreeLevel(args.key_type))
    tree_height = 10 if args.key_type == "unsigned" else 21
    if tree_height > max_level:
        raise SystemExit(
            f"Requested tree height {tree_height} exceeds maxTreeLevel={max_level} for key_type={args.key_type}"
        )
    if args.level > tree_height:
        raise SystemExit(f"--level must be <= tree_height ({tree_height})")
    print(f"Using maxTreeLevel={max_level} for key_type={args.key_type}, effective tree_height={tree_height}")

    octree_level = tree_height - args.level  # adjust for level-from-right

    if args.bits is None:
        bx_full, by_full, bz_full = map(int, cstone_sfc.getBoxDimBits(box_limits, args.key_type))
        downshift = max_level - tree_height
        bx, by, bz = bx_full - downshift, by_full - downshift, bz_full - downshift
        print(f"Derived bits from box dimensions: (bx, by, bz)=({bx}, {by}, {bz}), key_type={args.key_type}")
    else:
        bx, by, bz = args.bits
    validate_bits((bx, by, bz), tree_height)

    total_nodes = count_mixd_nodes_at_level(bx, by, bz, args.level)
    print(
        f"Total nodes at level={args.level}: {total_nodes}, traversing full valid key sequence "
        f"via increaseKey at level={args.level}."
    )

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    seen_iboxes: set[tuple[int, int, int, int, int, int]] = set()

    key = 0
    for node_idx in range(total_nodes):
        print(f"Visiting key {key} (octal: {oct(key)})...")
        ibox_arr = cstone_sfc.hilbertIBox(key, octree_level, bx, by, bz, args.key_type)
        ibox = (int(ibox_arr[0]), int(ibox_arr[1]), int(ibox_arr[2]), int(ibox_arr[3]), int(ibox_arr[4]), int(ibox_arr[5]))
        center, size = cstone_sfc.centerAndSize(ibox, box_limits, args.key_type)
        if not (size[0] <= 0 and size[1] <= 0 and size[2] <= 0):
            if ibox in seen_iboxes:
                raise SystemExit(
                    f"Error: duplicate ibox encountered for key={key}: {ibox}. "
                    "Each box should be seen only once."
                )
            seen_iboxes.add(ibox)
            xs.append(float(center[0]))
            ys.append(float(center[1]))
            zs.append(float(center[2]))
            if len(xs) % 100000 == 0:
                print(f"Processed {len(xs)} points...")

        # increaseKey position is counted from the left (0..max_level), while args.level is from the right.
        # increaseKey is used to create the keys of the leaf nodes of the octree at the specified level.
        next_key = int(cstone_sfc.increaseKey(key, octree_level, bx, by, bz, args.key_type))
        if node_idx == total_nodes - 1:
            break
        if next_key <= key:
            raise SystemExit(
                f"increaseKey terminated early at index={node_idx}, key={key} (octal: {oct(key)}), "
                f"expected total_nodes={total_nodes}."
            )
        key = next_key

    if len(seen_iboxes) != total_nodes:
        raise SystemExit(
            f"Traversal mismatch: expected {total_nodes} unique boxes, got {len(seen_iboxes)}."
        )

    if not xs:
        raise SystemExit("No points produced for the selected parameters.")

    print(f"First point: ({xs[0]}, {ys[0]}, {zs[0]}), Last point: ({xs[-1]}, {ys[-1]}, {zs[-1]}), Total points: {len(xs)}")

    points = list(zip(xs, ys, zs))
    unique_points = set(points)
    duplicate_points = len(points) - len(unique_points)
    consecutive_duplicates = 0
    for i in range(len(points) - 1):
        if points[i] == points[i + 1]:
            consecutive_duplicates += 1
    print(
        f"Duplicate diagnostics: total duplicates={duplicate_points}, "
        f"unique points={len(unique_points)}, consecutive duplicates={consecutive_duplicates}"
    )

    # Compute the largest distance between two consecutive points (excluding last-to-first)
    max_dist = 0.0
    max_idx = 0
    max_idx_next: int | None = None
    if len(xs) > 1:
        for i in range(len(xs) - 1):
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            dz = zs[i+1] - zs[i]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        max_idx_next = max_idx + 1
        print(
            f"Largest distance between two consecutive points (excluding last-to-first): {max_dist} "
            f"(between points {max_idx} and {max_idx_next})"
        )
    else:
        print("Largest distance between two consecutive points: N/A (only one point)")

    output_path = Path(args.save) if args.save else Path(f'plots/sfc_{args.lx}_{args.ly}_{args.lz}_level{args.level}_{args.key_type}.png')
    ext = output_path.suffix.lower() if output_path else ''

    if output_path and ext in {'.png', '.jpg', '.jpeg'}:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, linewidth=1.0, marker="o", markersize=2)
        # Mark start (green) and end (red)
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color="green", s=60, label="start")
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="red", s=60, label="end")
        # Highlight the two points with largest consecutive distance and connect them.
        if max_idx_next is not None:
            ax.scatter([xs[max_idx], xs[max_idx_next]], [ys[max_idx], ys[max_idx_next]], [zs[max_idx], zs[max_idx_next]],
                       color="orange", s=70, label="max-distance pair")
            ax.plot([xs[max_idx], xs[max_idx_next]], [ys[max_idx], ys[max_idx_next]], [zs[max_idx], zs[max_idx_next]],
                    color="orange", linewidth=2.0, linestyle="--")
        ax.legend()
        ax.set_title(
            f"MixD Hilbert curve centers (bx, by, bz)=({bx}, {by}, {bz}), key_type={args.key_type}, "
            f"level={args.level}"
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
            x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
            mode='markers',
            marker=dict(size=7, color='red', symbol='circle'),
            name='end'
        )
        max_pair_marker = None
        if max_idx_next is not None:
            max_pair_marker = go.Scatter3d(
                x=[xs[max_idx], xs[max_idx_next]], y=[ys[max_idx], ys[max_idx_next]], z=[zs[max_idx], zs[max_idx_next]],
                mode='markers+lines',
                marker=dict(size=7, color='orange', symbol='diamond'),
                line=dict(width=5, color='orange', dash='dash'),
                name='max-distance pair'
            )
        layout = go.Layout(
            title=f"MixD Hilbert curve centers (bx, by, bz)=({bx}, {by}, {bz}), key_type={args.key_type}, level={args.level}",
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                aspectmode='manual',
                aspectratio=dict(x=args.lx, y=args.ly, z=args.lz)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        plotly_data = [trace, start_marker, end_marker]
        if max_pair_marker is not None:
            plotly_data.append(max_pair_marker)
        fig = go.Figure(data=plotly_data, layout=layout)

        if output_path:
            html_path = output_path.with_suffix('.html')
            html_path.parent.mkdir(parents=True, exist_ok=True)
            pio.write_html(fig, file=str(html_path), auto_open=False, include_plotlyjs='embed')
            print(f"Saved interactive plot to {html_path}")
        else:
            print("Displaying interactively the plot doesn't work. Plese use --save with an `.html` suffix to save the plot as an HTML file.")


if __name__ == "__main__":
    main()
