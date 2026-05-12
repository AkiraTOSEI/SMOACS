#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.ext.matproj import MPRester


DEFAULT_CHEMSYS = ["Rb-I", "Ca-F", "Li-O"]
DEFAULT_THERMO_TYPES = ["GGA_GGA+U", "R2SCAN", "GGA_GGA+U_R2SCAN"]


def _parse_elements(chemsys: str) -> List[str]:
    parts = [p.strip() for p in chemsys.replace(",", "-").split("-") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid chemsys: {chemsys}")
    return parts


def _fetch_entries(api_key: str, elements: Sequence[str], thermo_types: Sequence[str]):
    with MPRester(api_key) as mpr:
        return mpr.get_entries_in_chemsys(
            list(elements),
            additional_criteria={"thermo_types": list(thermo_types)},
            inc_structure=False,
        )


def _save_convex_hull(entries, out_path: Path, show_unstable: bool) -> None:
    compat = MaterialsProject2020Compatibility(
        check_potcar_hash=False, check_potcar=False
    )
    proc = compat.process_entries(entries, clean=True)
    if not proc:
        print(f"[WARN] No compatible entries for {out_path.stem}. Skipping.")
        return
    phase = PhaseDiagram(proc)
    plotter = PDPlotter(phase, backend="matplotlib", show_unstable=show_unstable)
    plot = plotter.get_plot()
    fig = getattr(plot, "figure", plot)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _iter_chemsys(values: Iterable[str]) -> List[str]:
    items: List[str] = []
    for val in values:
        if not val:
            continue
        if "," in val and "-" not in val:
            items.extend([v.strip() for v in val.split(",") if v.strip()])
        else:
            items.append(val)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot convex hulls from Materials Project entries."
    )
    parser.add_argument(
        "--chemsys",
        nargs="+",
        default=DEFAULT_CHEMSYS,
        help="Chemical systems like Rb-I Ca-F Li-O (comma also allowed).",
    )
    parser.add_argument(
        "--api_key",
        "--mp_api_key",
        dest="api_key",
        default=os.environ.get("MP_API_KEY", ""),
        help="Materials Project API key (or set MP_API_KEY).",
    )
    parser.add_argument(
        "--out_dir",
        default="./convex_hulls",
        help="Output directory for PNG files.",
    )
    parser.add_argument(
        "--thermo_types",
        nargs="+",
        default=DEFAULT_THERMO_TYPES,
        help="Thermo types to include in MP query.",
    )
    parser.add_argument(
        "--show_unstable",
        action="store_true",
        help="Include unstable entries in the plot.",
    )
    args = parser.parse_args()

    api_key = args.api_key.strip()
    if not api_key:
        raise SystemExit("MP API key is required (use --api_key or MP_API_KEY).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chemsys_list = _iter_chemsys(args.chemsys)
    for chemsys in chemsys_list:
        elements = _parse_elements(chemsys)
        entries = _fetch_entries(api_key, elements, args.thermo_types)
        if not entries:
            print(f"[WARN] No entries for {chemsys}. Skipping.")
            continue
        label = "-".join(elements)
        out_path = out_dir / f"convex_hull_{label}.png"
        print(f"Building convex hull for {label} -> {out_path}")
        _save_convex_hull(entries, out_path, show_unstable=args.show_unstable)


if __name__ == "__main__":
    main()
