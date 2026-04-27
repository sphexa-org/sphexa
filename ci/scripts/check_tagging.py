import re
import sys
from pathlib import Path

import h5py
import numpy as np

# Check if tagging attributes are correctly reported in the full output file
def check_attributes(tag_setup_path: Path, full_output_path: Path) -> int:
    with h5py.File(tag_setup_path, "r") as fa, h5py.File(full_output_path, "r") as fb:
        attrs_a = set(fa.attrs.keys())
        attrs_b = set(fb.attrs.keys())

    missing_attrs = sorted(attrs_a - attrs_b)

    if not missing_attrs:
        print("All tagging attributes are present in the full output file.")
        return 0

    print("The full output file is missing tagging attributes.")
    print("\nMissing tagging attributes in the full output file:")
    print(f"  - {', '.join(missing_attrs)}")

    return 1


def _decode_attr_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    if hasattr(value, "tolist"):
        listed = value.tolist()
        if isinstance(listed, list):
            parts = []
            for item in listed:
                if isinstance(item, bytes):
                    parts.append(item.decode())
                else:
                    parts.append(str(item))
            return ",".join(parts)
        return str(listed)
    return str(value)


def _parse_field_list(value) -> list[str]:
    raw = _decode_attr_value(value)
    fields = [item.strip() for item in re.split(r"[\s,]+", raw) if item.strip()]
    return list(dict.fromkeys(fields))


def read_expected_subset_fields(tag_setup_path: Path) -> list[str]:

    with h5py.File(tag_setup_path, "r") as tag_file:
        if "f_subset" in tag_file.attrs:
            fields = _parse_field_list(tag_file.attrs["f_subset"])
            if fields:
                return fields
    return []


def read_subset_dataset_names(subset_output_path: Path) -> set[str]:
    names: set[str] = set()
    with h5py.File(subset_output_path, "r") as subset_file:

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                names.add(name.split("/")[-1])

        subset_file.visititems(visitor)
    return names

# Check if every field listed in f_subset attribute of the tag setup file is present as a dataset in the subset file
def check_fields(tag_setup_path: Path, subset_output_path: Path) -> int:
    expected = read_expected_subset_fields(tag_setup_path)
    if not expected:
        print("f_subset not present in tag setup file, no expected fields to check.")
        return 0

    available = read_subset_dataset_names(subset_output_path)
    missing = [field for field in expected if field not in available]

    if missing:
        print("Subset output file is missing required fields.")
        print(f"Expected from f_subset: {', '.join(expected)}.")
        print(f"Missing fields: {', '.join(missing)}.")
        return 1

    print("Subset output file contains all fields listed in f_subset.")
    return 0


def collect_ids(h5file: h5py.File) -> dict[str, set[int]]:
    """Collect integer values from every dataset named 'id'."""
    result: dict[str, set[int]] = {}

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == "id":
            values = np.asarray(obj[...]).reshape(-1)
            result[f"/{name}"] = {int(v) for v in values.tolist()}

    h5file.visititems(visitor)
    return result

# Check if every id value in the subset file exists in the full output file
def check_ids(full_output_path: Path, subset_output_path: Path) -> int:
    with h5py.File(full_output_path, "r") as full_file, h5py.File(subset_output_path, "r") as subset_file:
        full_ids_list = collect_ids(full_file)
        subset_ids_list = collect_ids(subset_file)

    if not full_ids_list:
        print(f"Error: no dataset named 'id' found in full output file: {full_output_path}.")
        return 2

    if not subset_ids_list:
        print(f"Error: no dataset named 'id' found in subset file: {subset_output_path}.")
        return 2

    full_ids = set().union(*full_ids_list.values())
    subset_ids = set().union(*subset_ids_list.values())

    missing = sorted(subset_ids - full_ids)
    if not missing:
        print("Every id in subset file exists in full output file.")
        return 0

    preview = ", ".join(str(v) for v in missing[:20])
    if len(missing) > 20:
        preview += ", ..."

    print("Subset file contains id values not present in full output file.")
    print(f"Missing ids in full output ({len(missing)} total): {preview}")
    return 1


def main(tag_setup_path: Path, dump_path: Path, subset_path: Path) -> int:
    status = 0

    status = max(status, check_attributes(tag_setup_path, dump_path))
    status = max(status, check_ids(dump_path, subset_path))
    status = max(status, check_fields(tag_setup_path, subset_path))
    # TODO: add check on iteration numbers compared to the tag setup file, if available
    return status


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(
            f"usage: {Path(sys.argv[0]).name} <tag_setup_or_restart_file> <dump_file> <subset_file>"
        )

    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
