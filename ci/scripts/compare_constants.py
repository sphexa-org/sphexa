import sys
from pathlib import Path

eps_zero = 1e-13
relative_tolerance = 1e-2
absolute_tolerance = 1e-6


def load_rows(path: Path):
    rows = []
    with path.open() as fh:
        for row in fh:
            cols = row.split()
            rows.append([float(p) for p in cols])
    return rows


def main(ref_path: Path, new_path: Path, abs_check_cols: list[int] | None = None, rel_tolerance: float = relative_tolerance, absolute_tolerance: float = absolute_tolerance):
    ref = load_rows(ref_path)
    new = load_rows(new_path)
    if len(ref) != len(new):
        print(f"Length mismatch: {len(ref)} (ref) vs {len(new)} (new)")
        sys.exit(2)
    above_tolerance = []
    abs_check_c = set(abs_check_cols or [])
    print(f"Absolute value check for columns: {sorted(abs_check_c)}")
    for idx in range(len(ref)):
        r, n = ref[idx], new[idx]
        if len(r) != len(n):
            print(f"Column mismatch on row {idx}: {len(r)} vs {len(n)}")
            continue
        for col, (vr, vn) in enumerate(zip(r, n)):
            diff = abs(vn - vr)
            den = max(abs(vr), eps_zero)
            rel = diff / den
            if col in abs_check_c: # use absolute difference
                if diff > absolute_tolerance:
                    above_tolerance.append((idx, col, vr, vn, diff, rel, 'abs'))
            else:
                if rel > rel_tolerance:
                    above_tolerance.append((idx, col, vr, vn, diff, rel, 'rel'))

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