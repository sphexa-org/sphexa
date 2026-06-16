import numpy as np
import sys
from pathlib import Path

eps_zero = 1e-13
relative_tolerance = 1e-2
absolute_tolerance = 1e-6


def main(
    ref_path: Path,
    new_path: Path,
    abs_check_cols: list[int] | None = None,
    rel_tolerance: float = relative_tolerance,
    absolute_tolerance: float = absolute_tolerance,
):
    ref = np.loadtxt(ref_path)
    new = np.loadtxt(new_path)
    if ref.shape != new.shape:
        print(f"Shape mismatch: {ref.shape} (ref) vs {new.shape} (new)")
        sys.exit(2)
    abs_check_c = set(abs_check_cols or [])
    print(f"Absolute value check for columns: {sorted(abs_check_c)}")
    abs_diffs = abs(new[:, 1:] - ref[:, 1:])
    denoms = np.maximum(np.abs(ref[:, 1:]), eps_zero)
    rel_diffs = abs_diffs / denoms
    ncols = abs_diffs.shape[1]

    # Define absolute and relative check column masks
    col_mask_abs = np.zeros(ncols, dtype=bool)
    cols_abs_check = np.array(
        [c - 1 for c in abs_check_c if 1 <= c <= ncols],
        dtype=int,
    )
    col_mask_abs[cols_abs_check] = True
    col_mask_rel = ~col_mask_abs

    # Determine where absolute or relative differences exceed tolerances
    abs_mask = (abs_diffs > absolute_tolerance) & col_mask_abs
    rel_mask = (rel_diffs > rel_tolerance) & col_mask_rel

    above_tolerance = []
    above_tolerance_kind = []

    for mask, kind in ((abs_mask, "abs"), (rel_mask, "rel")):
        rows, cols = np.where(mask)
        if rows.size == 0:
            continue
        above_tolerance.append(np.column_stack((
            rows,
            cols + 1,
            ref[rows, cols + 1],
            new[rows, cols + 1],
            abs_diffs[rows, cols],
            rel_diffs[rows, cols],
        )))
        above_tolerance_kind.append(np.full(rows.size, kind))

    if above_tolerance:
        above_tolerance = np.vstack(above_tolerance)
        above_tolerance_kind = np.concatenate(above_tolerance_kind)
    else:
        above_tolerance = np.empty((0, 6))
        above_tolerance_kind = np.empty((0,), dtype='<U3')

    if above_tolerance.shape[0] > 0:
        print("Rows exceeding tolerances (row, col, ref, new, abs, rel, error type):")
        for entry, kind in zip(above_tolerance, above_tolerance_kind):
            row, col, ref_val, new_val, abs_val, rel_val = entry
            print((
                int(row),
                int(col),
                float(ref_val),
                float(new_val),
                float(abs_val),
                float(rel_val),
                str(kind),
            ))
        sys.exit(1)
    else:
        print("Files agree within tolerances.")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(
            f"usage: {Path(sys.argv[0]).name} <reference> <new> "
            f"[comma-separated ignored columns]"
        )

    abs_cols_arg = sys.argv[3] if len(sys.argv) == 4 else ""
    if abs_cols_arg:
        try:
            abs_check_columns = [int(idx.strip()) for idx in
                                 abs_cols_arg.split(",") if idx.strip()]
        except ValueError as exc:
            sys.exit(f"Invalid absolute check column specification "
                     f"'{abs_cols_arg}': {exc}")
    else:
        abs_check_columns = []

    main(Path(sys.argv[1]), Path(sys.argv[2]),
         abs_check_cols=abs_check_columns)
