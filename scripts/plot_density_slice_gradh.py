#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scalar_attr(group, name, default=None):
    if name not in group.attrs:
        return default

    value = group.attrs[name]
    if np.ndim(value) == 0:
        return value.item()
    return np.asarray(value).flat[0]


def print_steps(h5file):
    print("Available steps:")
    print("hdf5 step".rjust(12), "iteration".rjust(12), "time".rjust(16))
    for key in sorted(h5file.keys(), key=step_sort_key):
        if not key.startswith("Step#"):
            continue
        step = h5file[key]
        iteration = scalar_attr(step, "iteration", "-")
        time = scalar_attr(step, "time", "-")
        print(key.replace("Step#", "").rjust(12), str(iteration).rjust(12), str(time).rjust(16))


def step_sort_key(name):
    if name.startswith("Step#"):
        try:
            return int(name[5:])
        except ValueError:
            pass
    return name


def open_step(h5file, step_number):
    key = f"Step#{step_number}"
    if key not in h5file:
        print(f"Step {step_number} not found in {h5file.filename}", file=sys.stderr)
        print_steps(h5file)
        sys.exit(1)
    return h5file[key]


def read_field(step, name):
    if name not in step:
        raise KeyError(f"Required field '{name}' is missing from {step.name}")
    return np.asarray(step[name])


def read_density(step):
    if "rho" in step:
        return np.asarray(step["rho"]), "rho"

    missing = [field for field in ("kx", "m", "xm") if field not in step]
    if missing:
        raise KeyError("Required field 'rho' is missing and density cannot be reconstructed "
                       f"because {', '.join(missing)} are missing")

    return np.asarray(step["kx"]) * np.asarray(step["m"]) / np.asarray(step["xm"]), "kx*m/xm"


def plot_density_slice(step, step_number, input_name, output_path, point_size):
    x = read_field(step, "x")
    y = read_field(step, "y")
    z = read_field(step, "z")
    h = read_field(step, "h")
    rho, rho_label = read_density(step)

    mask = np.isfinite(z) & np.isfinite(h) & (h > 0) & (np.abs(z) / h <= 2.0)
    if not np.any(mask):
        raise RuntimeError("No particles selected by abs(z) / h <= 2")

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")

    plot = ax.scatter(x[mask], y[mask], c=rho[mask], s=point_size, cmap="viridis", linewidths=0, rasterized=True)
    colorbar = fig.colorbar(plot, ax=ax)
    colorbar.set_label("density")

    time = scalar_attr(step, "time")
    title = f"{Path(input_name).name}, Step#{step_number}, Nslice={np.count_nonzero(mask)}"
    if time is not None:
        title += f", t={time:g}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path} using density field {rho_label}")


def plot_gradh_vs_x(step, step_number, input_name, output_path, point_size):
    x = read_field(step, "x")
    gradh = read_field(step, "gradh")

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.scatter(x, gradh, s=point_size, c="black", alpha=0.45, linewidths=0, rasterized=True)

    time = scalar_attr(step, "time")
    title = f"{Path(input_name).name}, Step#{step_number}, N={len(x)}"
    if time is not None:
        title += f", t={time:g}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("gradh")
    ax.grid(True, alpha=0.25)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def output_paths(input_file, step_number, output_dir):
    stem = Path(input_file).stem
    prefix = output_dir / f"{stem}_step{step_number}"
    return prefix.with_name(prefix.name + "_density_z0.png"), prefix.with_name(prefix.name + "_gradh_vs_x.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot an SPH-EXA density slice around z=0 and gradh versus x from an HDF5 output step."
    )
    parser.add_argument("file", help="SPH-EXA HDF5 output file")
    parser.add_argument("step", nargs="?", type=int, help="HDF5 step number, e.g. 0 for /Step#0")
    parser.add_argument("-p", "--print-steps", action="store_true", help="List available HDF5 step numbers and exit")
    parser.add_argument("-o", "--output-dir", default=".", help="Directory for generated PNG files")
    parser.add_argument("--point-size", type=float, default=1.0, help="Scatter point size")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.file, "r") as h5file:
        if args.print_steps:
            print_steps(h5file)
            return

        if args.step is None:
            print("Missing step number. Use -p to list available steps.", file=sys.stderr)
            sys.exit(2)

        density_png, gradh_png = output_paths(args.file, args.step, output_dir)
        step = open_step(h5file, args.step)
        plot_density_slice(step, args.step, args.file, density_png, args.point_size)
        plot_gradh_vs_x(step, args.step, args.file, gradh_png, args.point_size)


if __name__ == "__main__":
    main()
