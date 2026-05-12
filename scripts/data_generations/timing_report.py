#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _normalize_paths(out_path: Path) -> tuple[Path, Path]:
    if out_path.suffix == ".json":
        json_path = out_path
        csv_path = out_path.with_suffix(".csv")
    elif out_path.suffix == ".csv":
        csv_path = out_path
        json_path = out_path.with_suffix(".json")
    else:
        json_path = out_path.with_suffix(".json")
        csv_path = out_path.with_suffix(".csv")
    return json_path, csv_path


def write_timing_report(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = _normalize_paths(out_path)
    json_path.write_text(json.dumps(payload, indent=2))

    rows = []
    for key, value in payload.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                rows.append({"key": f"{key}.{sub_key}", "value": sub_val})
        else:
            rows.append({"key": key, "value": value})

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["key", "value"])
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["write_timing_report"]
