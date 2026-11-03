import json
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML configs") from exc
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise ValueError("Unsupported config format. Use .json or .yaml/.yml")


def add_force_pressure_arguments(parser):
    """Register the independent force/pressure CLI controls."""
    for term, coefficient in (("force", "F_ref"), ("pressure", "P_ref")):
        parser.add_argument(f"--use_{term}_loss", action="store_true")
        parser.add_argument(
            f"--{term}_loss_mode", choices=("log", "raw"), default="log"
        )
        parser.add_argument(f"--{coefficient}", type=float, default=1.0)
        parser.add_argument(f"--{term}_loss_clip", type=float, default=1e-7)


def build_force_pressure_config(args):
    """Resolve and validate CLI values before loading any model."""
    import math

    config = {}
    for term, coefficient in (("force", "F_ref"), ("pressure", "P_ref")):
        enabled = getattr(args, f"use_{term}_loss")
        mode = getattr(args, f"{term}_loss_mode")
        weight = getattr(args, coefficient)
        clip = getattr(args, f"{term}_loss_clip")
        if mode not in ("log", "raw"):
            raise ValueError(f"{term}_loss_mode must be log or raw")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{coefficient} must be finite and >= 0")
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError(f"{term}_loss_clip must be finite and > 0")
        config.update(
            {
                f"use_{term}_loss": enabled,
                f"{term}_loss_mode": mode,
                coefficient: weight,
                f"{term}_loss_clip": clip,
            }
        )
    return config
