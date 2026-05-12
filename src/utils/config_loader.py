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
