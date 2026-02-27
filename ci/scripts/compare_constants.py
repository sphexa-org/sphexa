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
    for row_idx, abs_row in enumerate(abs_diffs):
        for col_idx, abs_diff in enumerate(abs_row):
            if col_idx in abs_check_c and abs_diff > absolute_tolerance:
                above_tolerance.append((row_idx, col_idx+1, ref[row_idx, col_idx+1], new[row_idx, col_idx+1], abs_diff, rel_diffs[row_idx, col_idx], 'abs'))
            elif col_idx not in abs_check_c and rel_diffs[row_idx, col_idx] > rel_tolerance:
                above_tolerance.append((row_idx, col_idx+1, ref[row_idx, col_idx+1], new[row_idx, col_idx+1], abs_diff, rel_diffs[row_idx, col_idx], 'rel'))

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