from pathlib import Path

from periodictable import elements


def get_master_dir(project_name="SMOACS") -> str:
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if parent.name == project_name:
            return str(parent)
    raise ValueError(f"'{project_name}' not found in {current}")


def create_element_list() -> list[str]:
    """Return a list of the first 98 valid chemical element symbols."""
    return [elements[i].symbol for i in range(1, 99)]
