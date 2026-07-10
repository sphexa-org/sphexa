#!/usr/bin/env python3
"""Print a timing comparison table between two sphexa log files.

Usage:
    timing_summary.py <mixd_log> <dev_log> <test> <backend>

Lines parsed from logs have the form:
    # <category>: <value>s
"""

import re
import sys
from collections import OrderedDict


_TOTAL_TIME_RE = re.compile(r"^Total execution time of \d+ iterations of .+")


def _normalize_category(cat: str) -> str:
    """Shorten the verbose 'Total execution time ...' line to a fixed label."""
    if _TOTAL_TIME_RE.match(cat):
        return "Total time reported"
    return cat


def parse_log(path: str) -> "OrderedDict[str, float]":
    totals: OrderedDict[str, float] = OrderedDict()
    pattern = re.compile(r"^# (.+): ([0-9][^ ]*)s")
    with open(path) as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                cat = _normalize_category(m.group(1))
                val = float(m.group(2))
                totals[cat] = totals.get(cat, 0.0) + val
    return totals


def main() -> None:
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <mixd_log> <dev_log> <test> <backend>", file=sys.stderr)
        sys.exit(1)

    mixd_log, dev_log, test, backend = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    mixd = parse_log(mixd_log)
    dev  = parse_log(dev_log)

    # Preserve order: MixD categories first, then any extras from dev
    categories = list(mixd.keys()) + [k for k in dev if k not in mixd]

    title = f"Timing Comparison: MixD vs develop  [{test.upper()} / {backend.upper()}]  (summed over all steps)"
    sep   = "─" * (45 + 4 * 13)

    print()
    print(f"=== {title} ===")
    print(sep)
    print(f"{'Category':<45} {'MixD (s)':>12} {'Develop (s)':>12} {'Diff (s)':>12} {'Diff (%)':>12}")
    print(sep)

    for cat in categories:
        m = mixd.get(cat, 0.0)
        d = dev.get(cat, 0.0)
        diff   = m - d
        pct    = (diff / d * 100) if d != 0.0 else float("nan")
        pct_str = f"{pct:+.1f}" if d != 0.0 else "  n/a"
        print(f"{cat:<45} {m:>12.3f} {d:>12.3f} {diff:>+12.3f} {pct_str:>12}")

    print(sep)


if __name__ == "__main__":
    main()
