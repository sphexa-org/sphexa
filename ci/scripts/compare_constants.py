import numpy as np
import sys
from pathlib import Path

eps_zero = 1e-13
relative_tolerance = 1e-2
absolute_tolerance = 1e-6

def main(ref_path: Path, new_path: Path, abs_check_cols: list[int] | None = None, rel_tolerance: float = relative_tolerance, absolute_tolerance: float = absolute_tolerance):
    ref = np.loadtxt(ref_path)
    new = np.loadtxt(new_path)
    if ref.shape != new.shape:
        print(f"Shape mismatch: {ref.shape} (ref) vs {new.shape} (new)")
        sys.exit(2)
    above_tolerance = []
    abs_check_c = set(abs_check_cols or [])
    print(f"Absolute value check for columns: {sorted(abs_check_c)}")
    abs_diffs = abs(new[:,1:] - ref[:,1:])
    denoms = np.maximum(np.abs(ref[:,1:]), eps_zero)
    rel_diffs = abs_diffs/denoms
    for row in range(abs_diffs.shape[0]):
        for col in range(abs_diffs.shape[1]):
             if col in abs_check_c and abs_diffs[row, col] > absolute_tolerance:
                above_tolerance.append((row, col+1, ref[row, col+1], new[row, col+1], abs_diffs[row, col], rel_diffs[row, col], 'abs'))
             elif col not in abs_check_c and rel_diffs[row, col] > rel_tolerance:
                above_tolerance.append((row, col+1, ref[row, col+1], new[row, col+1], abs_diffs[row, col], rel_diffs[row, col], 'rel'))

    if above_tolerance:
        print("Rows exceeding tolerances (row, col, ref, new, abs, rel, error type):")
        for entry in above_tolerance:
            print(entry)
        sys.exit(1)
    else:
        print("Files agree within tolerances.")
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(
            f"usage: {Path(sys.argv[0]).name} <reference> <new> [comma-separated ignored columns]"
        )

    abs_cols_arg = sys.argv[3] if len(sys.argv) == 4 else ""
    if abs_cols_arg:
        try:
            abs_check_columns = [int(idx.strip()) for idx in abs_cols_arg.split(",") if idx.strip()]
        except ValueError as exc:
            sys.exit(f"Invalid absolute check column specification '{abs_cols_arg}': {exc}")
    else:
        abs_check_columns = []

    main(Path(sys.argv[1]), Path(sys.argv[2]), abs_check_cols=abs_check_columns)