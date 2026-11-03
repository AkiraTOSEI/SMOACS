#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo 'Usage: bash run_mattersim.sh section_2_3|section_2_4 EXPERIMENT_NAME' >&2
  exit 2
fi
SECTION="$1"
EXPERIMENT="$2"
if [[ "$SECTION" != section_2_3 && "$SECTION" != section_2_4 ]]; then
  echo "Unsupported section: $SECTION" >&2
  exit 2
fi
if [[ ! "$EXPERIMENT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo 'Experiment must be a single directory name' >&2
  exit 2
fi
: "${MP_API_KEY:?Set MP_API_KEY in your environment before running}"
REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$REPRO_ROOT/.." && pwd)"
WORK_ROOT="$REPRO_ROOT/$SECTION/mattersim"
RESULT_DIR="$WORK_ROOT/results/$EXPERIMENT"
SCRIPT="$REPO_ROOT/scripts/calculate_mattersim_convex_hull_energy.py"
PYTHON_BIN="${PYTHON:-python}"
if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing $SCRIPT; place reproducibility/ directly under the repository root" >&2
  exit 1
fi
if [[ ! -f "$REPRO_ROOT/$SECTION/screening_logs/${EXPERIMENT}_mattersim_inputs.csv" ]]; then
  echo 'Input manifest is missing; run prepare_mattersim_inputs.py for this section/experiment first' >&2
  exit 1
fi
shopt -s nullglob
POSCARS=("$WORK_ROOT/to_relax_poscars/$EXPERIMENT/"*.vasp)
if [[ ${#POSCARS[@]} == 0 ]]; then
  echo 'No prepared POSCARs; run prepare_mattersim_inputs.py first' >&2
  exit 1
fi
for directory in results relaxed_poscars DFT_POSCARS cache reference; do
  if [[ -e "$WORK_ROOT/$directory/$EXPERIMENT" ]]; then
    echo "Output already exists: $WORK_ROOT/$directory/$EXPERIMENT" >&2
    exit 1
  fi
done

steps_max=400
if [[ "${SMOACS_SMOKE_TEST:-0}" == 1 ]]; then
  steps_max=10
fi

# Record installed versions and the exact local checkpoint before any calculation.
"$PYTHON_BIN" - "$RESULT_DIR" "$SCRIPT" "$steps_max" <<'PY'
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

result_dir, script = map(Path, sys.argv[1:3])
steps_max = int(sys.argv[3])
checkpoint = Path.home() / ".local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth"
record = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {name: importlib.metadata.version(name) for name in (
        "mattersim", "torch", "numpy", "pymatgen", "ase", "mp-api", "atomate2"
    )},
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    "steps_max": steps_max,
    "stages": ["prefetch_MP", "MatterSim_phase1", "prefetch_Alex", "MatterSim_phase2"],
}
result_dir.mkdir(parents=True)
with (result_dir / "run_environment.json").open("x") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY

# The upstream script writes logs/timing_logs/mp_relaxed relative to the cwd.
cd "$RESULT_DIR"
COMMON_ARGS=(
  -p_dir "$WORK_ROOT/to_relax_poscars/$EXPERIMENT"
  -r_dir "$WORK_ROOT/relaxed_poscars/$EXPERIMENT"
  -rr_dir "$RESULT_DIR"
  -s_dir "$WORK_ROOT/DFT_POSCARS/$EXPERIMENT"
  --cache_dir "$WORK_ROOT/cache/$EXPERIMENT"
  --global_relax_dir "$WORK_ROOT/reference/$EXPERIMENT"
  --alexandria_base_url https://alexandria.icams.rub.de/pbe
  --steps_max "$steps_max"
)
for stage in prefetch_MP MatterSim_phase1 prefetch_Alex MatterSim_phase2; do
  API_ARGS=()
  if [[ "$stage" == prefetch_MP ]]; then
    API_ARGS=(--mp_api_key "$MP_API_KEY")
  fi
  "$PYTHON_BIN" "$SCRIPT" --stage "$stage" "${COMMON_ARGS[@]}" "${API_ARGS[@]}" \
    2>&1 | tee "$RESULT_DIR/$stage.log"
done
