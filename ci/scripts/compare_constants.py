import sys
from pathlib import Path

eps_zero = 1e-30
relative_tolerance = 1e-2


def load_rows(path: Path):
    rows = []
    with path.open() as fh:
        for row in fh:
            cols = row.split()
            rows.append([float(p) for p in cols])
    return rows


def main(ref_path: Path, new_path: Path):
    ref = load_rows(ref_path)
    new = load_rows(new_path)
    if len(ref) != len(new):
        print(f"Length mismatch: {len(ref)} (ref) vs {len(new)} (new)")
        sys.exit(2)
    above_tolerance = []
    for idx in range(len(ref)):
        r, n = ref[idx], new[idx]
        if len(r) != len(n):
            print(f"Column mismatch on row {idx}: {len(r)} vs {len(n)}")
            continue
        for col, (vr, vn) in enumerate(zip(r, n)):
            diff = abs(vn - vr)
            den = max(abs(vr), eps_zero)
            rel = diff / den
            if rel > relative_tolerance:
                above_tolerance.append((idx, col, vr, vn, diff, rel))
    if above_tolerance:
        print("Rows exceeding tolerances (row, col, ref, new, abs, rel):")
        for entry in above_tolerance:
            print(entry)
        sys.exit(1)
    else:
        print("Files agree within tolerances.")
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <reference> <new>")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
