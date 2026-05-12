import matplotlib.pyplot as plt
import numpy as np
import os
import re

def read_data_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    parsed = [list(map(float, line.split())) for line in lines]
    return np.array(parsed)

BACKEND = 'cpu'  # Change to 'gpu' if needed
STEPS = 200
N = 50
if BACKEND == 'cpu':
    RANKS = 4
    CORES = 72
elif BACKEND == 'gpu':
    RANKS = 1
    CORES = 1

TEST='kelvin' #'windshock' or 'kelvin'

# normalize and validate TEST value (accept common variants)
TEST = TEST.strip().lower()
if TEST in ('wind_shock', 'wind-shock', 'wind shock'):
    TEST = 'windshock'
if TEST not in ('windshock', 'kelvin'):
    raise ValueError("TEST must be either 'windshock' or 'kelvin' (case-insensitive)")

# File names (adjust paths if needed)
mixDConstants = f'/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/build_{BACKEND}/constants_{TEST}_r{RANKS}_c{CORES}_s{STEPS}_n{N}_{BACKEND}.txt'
developConstants = f'/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa-develop/build_{BACKEND}/constants_{TEST}_r{RANKS}_c{CORES}_s{STEPS}_n{N}_{BACKEND}.txt'

base1 = os.path.basename(mixDConstants)
base2 = os.path.basename(developConstants)
if base1 != base2:
    raise ValueError(f"Filenames must match but differ: {base1!r} != {base2!r}")

for p in (mixDConstants, developConstants):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Expected file not found: {p}")

# Read data from files
mixDData = read_data_from_file(mixDConstants)
developData = read_data_from_file(developConstants)

# Extract relevant columns
x = mixDData[:, 0]
y1 = mixDData[:, -2 if TEST == 'windshock' else -1]
y2 = developData[:, -2 if TEST == 'windshock' else -1]

# Compute absolute and relative differences
abs_diff = np.abs(y1 - y2)
with np.errstate(divide='ignore', invalid='ignore'):
    rel_diff = abs_diff / np.where(np.abs(y2) > 0, np.abs(y2), np.nan)

# Summary statistics
abs_mean = np.nanmean(abs_diff)
abs_max = np.nanmax(abs_diff)
rel_mean = np.nanmean(rel_diff)
rel_max = np.nanmax(rel_diff)

# Final-point differences
final_abs = abs_diff[-1]
final_rel = rel_diff[-1]

# Add a legend entry showing absolute and relative differences (mean, max, final)
summary_label = (
    f"{BACKEND.upper()} — ranks: {RANKS}, cores: {CORES}\n"
    f"Abs diff — mean: {abs_mean:.3e}, max: {abs_max:.3e}, final: {final_abs:.3e}\n"
    f"Rel diff — mean: {rel_mean:.3e}, max: {rel_max:.3e}, final: {final_rel:.3e}"
)

plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='MixD')
plt.plot(x, y2, label='develop')
plt.xlabel('Iteration step')
plt.ylabel('Surviving Fraction' if TEST == 'windshock' else 'Growth-rate')
plt.title('Surviving Fraction per iteration step (139719 particles)' if TEST == 'windshock' else 'Growth-rate per iteration step (24M particles)')
# plt.xticks(x)
plt.grid(True)
plt.legend(title=summary_label, loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()
output_file = f'{TEST}_comparison_{STEPS}_{BACKEND}_{RANKS}_{CORES}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'Saved figure to {output_file}')