"""Compare total numHalos between MixD and develop profile.h5 files."""

import argparse
import h5py
import numpy as np
import os

parser = argparse.ArgumentParser(description="Compare numHalos between MixD and develop profile.h5.")
parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
parser.add_argument("--test",    choices=["windshock", "kelvin"], default="kelvin")
parser.add_argument("--steps",   type=int, default=3)
parser.add_argument("--n",       type=int, default=50)
parser.add_argument("--ranks",   type=int, default=None, help="Override MPI rank count")
parser.add_argument("--cores",   type=int, default=None, help="Override OMP thread count")
args = parser.parse_args()

BACKEND = args.backend
TEST    = args.test
STEPS   = args.steps
N       = args.n

if args.ranks is not None:
    RANKS = args.ranks
elif BACKEND == "cpu":
    RANKS = 16
else:
    RANKS = 1

if args.cores is not None:
    CORES = args.cores
elif BACKEND == "cpu":
    CORES = 18
else:
    CORES = 1

MIXD_ROOT = "/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa"
DEV_ROOT  = "/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa-develop"
RUN_DIR   = f"{TEST}_r{RANKS}_c{CORES}_s{STEPS}_n{N}_{BACKEND}"

mixd_h5 = os.path.join(MIXD_ROOT, f"build_{BACKEND}", RUN_DIR, "profile.h5")
dev_h5  = os.path.join(DEV_ROOT,  f"build_{BACKEND}", RUN_DIR, "profile.h5")


def find_dataset(path, dataset_name):
    """Return the first dataset named `dataset_name` found in any Step group."""
    with h5py.File(path, "r") as f:
        for group_name in f:
            grp = f[group_name]
            if dataset_name in grp:
                return np.array(grp[dataset_name])
    raise KeyError(f"Dataset '{dataset_name}' not found in {path}")


mixd_halos = find_dataset(mixd_h5, "numHalos")
dev_halos  = find_dataset(dev_h5,  "numHalos")

mixd_sum = int(mixd_halos.sum())
dev_sum  = int(dev_halos.sum())
ratio    = mixd_sum / dev_sum if dev_sum != 0 else float("nan")

print(f"Run:     {RUN_DIR}")
print(f"MixD    numHalos sum : {mixd_sum:,}")
print(f"Develop numHalos sum : {dev_sum:,}")
print(f"Ratio   MixD / Dev   : {ratio:.6f}")
