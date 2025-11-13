import matplotlib.pyplot as plt
import numpy as np

def read_data_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    parsed = [list(map(float, line.split())) for line in lines]
    return np.array(parsed)

# File names (adjust paths if needed)
mixDConstants = '/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/build_cpu/constants_windshock_r4_c71_d100_50_n50_cpu.txt'
developConstants = '/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa-develop/build_cpu/constants_windshock_r4_c71_d100_50_n50_cpu.txt'

# Read data from files
mixDData = read_data_from_file(mixDConstants)
developData = read_data_from_file(developConstants)

# Extract relevant columns
x = mixDData[:, 0]
y1 = mixDData[:, -2]
y2 = developData[:, -2]

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
    f"Abs diff — mean: {abs_mean:.3e}, max: {abs_max:.3e}, final: {final_abs:.3e}\n"
    f"Rel diff — mean: {rel_mean:.3e}, max: {rel_max:.3e}, final: {final_rel:.3e}"
)

plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='MixD')
plt.plot(x, y2, label='develop')
plt.xlabel('Iteration step')
plt.ylabel('Surviving Fraction')
plt.title('Surviving Fraction per iteration step (139719 particles)')
# plt.xticks(x)
plt.grid(True)
plt.legend(title=summary_label, loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()
output_file = 'surviving_fraction_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'Saved figure to {output_file}')