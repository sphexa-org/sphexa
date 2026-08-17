#!/usr/bin/env python3

"""Distributed density probability-density-function tool.

Python port of main/src/analytical_solutions/turbulence/density_pdf.cpp.
The input HDF5 (H5Part) file may be too large to fit into the memory of a
single process, so it is opened with a parallel h5py driver and each MPI
rank reads only its contiguous slice of the particle arrays. The per-rank
histograms are reduced onto rank 0, which writes the final PDF.

Usage example:
    mpirun -np 4 python density_pdf.py --file dump.h5 -o density_pdf.txt
"""

import os
from argparse import ArgumentParser

import h5py
import numpy as np
from mpi4py import MPI


def partition_range(R, i, N):
    """Balanced contiguous split of R elements onto N ranks, rank i.

    Exact replica of sphexa::partitionRange in ifile_io_hdf5.cpp:138-154.
    """
    s = R // N
    r = R % N
    if i < r:
        start = (s + 1) * i
        end = start + s + 1
    else:
        start = (s + 1) * r + s * (i - r)
        end = start + s
    return start, end


def select_step_group(file, step):
    """Return the Step#<step> group, mirroring H5PartSetStep semantics.

    step < 0 selects the last step present in the file.
    """
    step_names = [n for n in file.keys() if n.startswith("Step#")]
    if not step_names:
        raise RuntimeError("Input file does not contain any Step# groups")
    indices = sorted(int(n.split("#", 1)[1]) for n in step_names)
    if step < 0:
        step = indices[-1]
    name = "Step#%d" % step
    if name not in file:
        raise KeyError("Step %s not found in file (available: %s)" % (step, indices))
    return file[name]


def read_local_field(step_group, key, first_index, last_index):
    """Read only the [first_index, last_index) slice of a 1D dataset."""
    if first_index >= last_index:
        return np.empty(0, dtype=step_group[key].dtype)
    return np.array(step_group[key][first_index:last_index])


def compute_probability_distribution(data, reference_value, bin_count, bin_start, bin_end):
    """Histogram of log(data / reference_value) over (bin_start, bin_end].

    Vectorized replacement of computeProbabilityDistribution in
    density_pdf.hpp:30-48, preserving the half-open (lower, upper] bin
    semantics used there.
    """
    if data.size == 0:
        return np.zeros(bin_count, dtype=np.float64)

    np.divide(data, reference_value, out=data)
    np.log(data, out=data)

    bin_size = (bin_end - bin_start) / bin_count
    edges = bin_start + bin_size * np.arange(bin_count + 1)

    # np.digitize with right=True returns j for edges[j-1] < x <= edges[j].
    # Indices 0 and bin_count+1 correspond to values out of range and are
    # discarded, exactly matching the std::count_if predicate in the C++.
    indices = np.digitize(data, edges, right=True)
    in_range = (indices >= 1) & (indices <= bin_count)
    counts = np.bincount(indices[in_range], minlength=bin_count + 2)
    return counts[1:bin_count + 1].astype(np.float64)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    parser = ArgumentParser(description="Distributed density PDF from an SPH-EXA HDF5 file")
    parser.add_argument("--file", required=True, help="input HDF5 (H5Part) file")
    parser.add_argument("-n", type=int, default=50, help="number of bins (default: 50)")
    parser.add_argument("-s", type=int, default=-1, dest="step", help="snapshot index, -1 = last (default: -1)")
    parser.add_argument("-o", default="density_pdf.txt", help="output file (default: density_pdf.txt)")
    parser.add_argument("--sph", default="std", help="SPH flavour: 'std' or other (default: std)")
    parser.add_argument("--min", type=float, default=-8.0, help="log-density range minimum (default: -8.0)")
    parser.add_argument("--max", type=float, default=6.0, help="log-density range maximum (default: 6.0)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        if rank == 0:
            print("Please provide a existing file: no file found at %s" % args.file)
        comm.Abort(1)

    file = h5py.File(args.file, "r", driver="mpio", comm=comm)
    step_group = select_step_group(file, args.step)

    # Pick a field to determine the global particle count. Both code paths
    # below require "rho" (std) or "kx" (non-std); prefer rho when present.
    count_key = "rho" if args.sph == "std" else "kx"
    global_n = int(step_group[count_key].shape[0])

    first_index, last_index = partition_range(global_n, rank, num_ranks)
    local_n = last_index - first_index

    if rank == 0:
        print("Density-PDF: local particles: %d \t global particles: %d" % (local_n, global_n))

    if args.sph == "std":
        rho = read_local_field(step_group, "rho", first_index, last_index).astype(np.float32, copy=False)
    else:
        kx = read_local_field(step_group, "kx", first_index, last_index).astype(np.float32, copy=False)
        xm = read_local_field(step_group, "xm", first_index, last_index).astype(np.float32, copy=False)
        m = read_local_field(step_group, "m", first_index, last_index).astype(np.float32, copy=False)
        rho = kx * m / xm
        del kx, xm, m

    if rho.size != local_n:
        raise RuntimeError("rho length doesn't match local count: %d\t%d" % (rho.size, local_n))

    file.close()

    local_total_density = float(np.sum(rho, dtype=np.float64)) if local_n > 0 else 0.0
    print("rank %d, local average  density: %f" % (rank, local_total_density / local_n if local_n > 0 else float("nan")))

    reference_density = comm.allreduce(local_total_density, op=MPI.SUM) / global_n

    if rank == 0:
        print("starting PDF calculation with reference density %f" % reference_density)

    bins = compute_probability_distribution(rho, reference_density, args.n, args.min, args.max)
    reduced_bins = comm.reduce(bins, op=MPI.SUM, root=0)

    if rank == 0:
        bin_size = (args.max - args.min) / args.n
        reduced_bins = np.asarray(reduced_bins, dtype=np.float64)
        reduced_bins /= global_n * bin_size
        first_middle = args.min + 0.5 * bin_size

        with open(args.o, "w") as out_file:
            # header line containing metadata
            out_file.write("%d %g %g\n" % (args.n, bin_size, reference_density))
            for i in range(args.n):
                bin_center = bin_size * i + first_middle
                out_file.write("%g %g\n" % (bin_center, reduced_bins[i]))

        print("Calculated PDF for %d particles in %d bins." % (global_n, args.n))


if __name__ == "__main__":
    main()
