import os
from pathlib import Path

from periodictable import elements


def get_master_dir(project_name: str | None = None) -> str:
    env_root = os.environ.get("SMOACS_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"SMOACS_ROOT does not exist: {root}")
        return str(root)

    expected_names = [project_name] if project_name else ["SMOACS"]
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if parent.name in expected_names or (
            (parent / "pyproject.toml").exists() and (parent / "src").is_dir()
        ):
            return str(parent)

    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "pyproject.toml").exists() and (package_root / "src").is_dir():
        return str(package_root)
    raise ValueError(f"SMOACS repository root not found from {current}")


def create_element_list() -> list[str]:
    """Return a list of the first 98 valid chemical element symbols."""
    return [elements[i].symbol for i in range(1, 99)]
