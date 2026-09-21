"""Read VASP templates in an explicitly recorded order when one is supplied."""

import glob
import os
from pathlib import Path


def get_structure_paths(dir_path):
    paths = glob.glob(os.path.join(dir_path, "*.vasp"))
    order_path = Path(dir_path) / "template_order.txt"
    if not order_path.exists():
        return paths

    names = order_path.read_text(encoding="utf-8").splitlines()
    if not names or any(
        not name
        or name != name.strip()
        or name != Path(name).name
        or "\\" in name
        or not name.endswith(".vasp")
        for name in names
    ):
        raise ValueError(f"{order_path}: expected one .vasp basename per line")
    if len(names) != len(set(names)):
        raise ValueError(f"{order_path}: duplicate template filenames")

    actual_names = {Path(path).name for path in paths}
    missing = set(names) - actual_names
    unexpected = actual_names - set(names)
    if missing or unexpected:
        raise ValueError(
            f"{order_path}: template mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    ordered_paths = [os.path.join(dir_path, name) for name in names]
    for path in ordered_paths:
        if not Path(path).is_file():
            raise ValueError(f"{order_path}: template is not a file: {path}")
    return ordered_paths
