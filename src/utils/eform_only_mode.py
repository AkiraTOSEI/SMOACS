from __future__ import annotations

from typing import Any, Dict, Optional

EFORM_ONLY_TASK = "eform_only"
LEGACY_EFORM_ONLY_TASK = "bg_and_eform1"


def build_eform_only_alignn_config(
    alignn_weights_dir: Optional[str],
) -> Dict[str, Any]:
    return {
        "predictors": [
            {
                "name": "eform",
                "type": "alignn",
                "checkpoint": "mp_e_form_alignnn",
                "params": {"weights_dir": alignn_weights_dir},
            }
        ],
        "loss": {
            "terms": [
                {
                    "name": "loss_eform",
                    "expr": "pred.eform",
                    "weight": 1.0,
                }
            ]
        },
        "output": {
            "primary_metrics": {
                "bandgap_onehot": "pred.eform",
                "ef_onehot": "pred.eform",
            }
        },
    }


def validate_eform_only_args(
    task: str,
    target_space_group_no: int,
    target_val: Optional[Any],
) -> None:
    if task != EFORM_ONLY_TASK:
        return
    if target_space_group_no <= 1:
        raise ValueError("--task eform_only requires --space_group > 1.")
    if target_val is not None:
        raise ValueError("--task eform_only must not specify --target_val.")
