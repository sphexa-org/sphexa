import sys
import h5py
from pathlib import Path

def collect_attributes(h5file: h5py.File) -> dict[str, set[str]]:
    """Collect attribute names for root, groups and datasets."""
    result: dict[str, set[str]] = {"/": set(h5file.attrs.keys())}

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        path = f"/{name}" if name else "/"
        result[path] = set(obj.attrs.keys())

    h5file.visititems(visitor)
    return result


def compare_attributes(file_a: str, file_b: str) -> int:
    with h5py.File(file_a, "r") as fa, h5py.File(file_b, "r") as fb:
        attrs_a = collect_attributes(fa)
        attrs_b = collect_attributes(fb)

    missing_paths: list[str] = []
    missing_attrs: list[tuple[str, list[str]]] = []

    for path, names_a in sorted(attrs_a.items()):
        if path not in attrs_b:
            if names_a:
                missing_paths.append(path)
            continue

        names_b = attrs_b[path]
        missing = sorted(names_a - names_b)
        if missing:
            missing_attrs.append((path, missing))

    if not missing_paths and not missing_attrs:
        print("OK: all attributes from the tag setup file are present in the dump file")
        sys.exit(0)

    print("FAILED: the dump file is missing attributes from the tag setup file")

    if missing_paths:
        print("\nMissing objects in the dump file (therefore their attributes are missing):")
        for path in missing_paths:
            print(f"  - {path}")

    if missing_attrs:
        print("\nMissing attribute names in the dump file:")
        for path, attrs in missing_attrs:
            print(f"  - {path}: {', '.join(attrs)}")

    sys.exit(1)


def main(ref_path: Path, new_path: Path):
    with h5py.File(ref_path, "r") as fa, h5py.File(new_path, "r") as fb:
        attrs_a = collect_attributes(fa)
        attrs_b = collect_attributes(fb)

    missing_paths: list[str] = []
    missing_attrs: list[tuple[str, list[str]]] = []

    for path, names_a in sorted(attrs_a.items()):
        if path not in attrs_b:
            if names_a:
                missing_paths.append(path)
            continue

        names_b = attrs_b[path]
        missing = sorted(names_a - names_b)
        if missing:
            missing_attrs.append((path, missing))

    if not missing_paths and not missing_attrs:
        print("OK: all attributes from the tag setup file are present in the dump file")
        sys.exit(0)

    print("FAILED: the dump file is missing attributes from the tag setup file")

    if missing_paths:
        print("\nMissing objects in the dump file (therefore their attributes are missing):")
        for path in missing_paths:
            print(f"  - {path}")

    if missing_attrs:
        print("\nMissing attribute names in the dump file:")
        for path, attrs in missing_attrs:
            print(f"  - {path}: {', '.join(attrs)}")

    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <tag_setup_file> <dump_file>")
    main(Path(sys.argv[1]), Path(sys.argv[2]))