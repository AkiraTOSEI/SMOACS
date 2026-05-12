#!/usr/bin/env bash
set -euo pipefail

# Prepare SMOACS POSCAR outputs for MatterSim relaxation and convex-hull workflows.
# Input layout:
#   results/<experiment_name>/poscar/*.vasp
# Output layout:
#   <out-root>/<experiment_name>/*__<experiment_name>.vasp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${REPO_ROOT}/results"
DEST_ROOT="${SCRIPT_DIR}/to_relax_poscars_original"
MAX_COPIES_PER_CONDITION=""
CONDITION=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [--results-root DIR] [--condition NAME] [--out-root DIR] [--max-copies N]

Options:
  --results-root DIR  Directory containing SMOACS result directories.
                      Default: ${REPO_ROOT}/results
  --condition NAME    Process one SMOACS result directory under --results-root.
                      If omitted, all directories with a poscar/ child are processed.
  --out-root DIR      Output root for MatterSim proposal POSCARs.
                      Default: ${SCRIPT_DIR}/to_relax_poscars_original
  --max-copies N  Maximum number of files to copy per condition directory.
                  If omitted, all .vasp files are copied.
  -h, --help      Show this help.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --results-root|--source-root)
      if [[ "$#" -lt 2 ]]; then
        echo "ERROR: $1 requires a directory argument" >&2
        exit 1
      fi
      SRC_ROOT="$2"
      shift 2
      ;;
    --condition)
      if [[ "$#" -lt 2 ]]; then
        echo "ERROR: --condition requires a result directory name" >&2
        exit 1
      fi
      CONDITION="$2"
      shift 2
      ;;
    --out-root)
      if [[ "$#" -lt 2 ]]; then
        echo "ERROR: --out-root requires a directory argument" >&2
        exit 1
      fi
      DEST_ROOT="$2"
      shift 2
      ;;
    --max-copies)
      if [[ "$#" -lt 2 ]]; then
        echo "ERROR: --max-copies requires an integer argument" >&2
        exit 1
      fi
      MAX_COPIES_PER_CONDITION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${MAX_COPIES_PER_CONDITION}" ]]; then
  if ! [[ "${MAX_COPIES_PER_CONDITION}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --max-copies must be a positive integer: ${MAX_COPIES_PER_CONDITION}" >&2
    exit 1
  fi
fi

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "ERROR: SMOACS results root directory not found: ${SRC_ROOT}" >&2
  exit 1
fi

declare -a CONDITIONS=()
declare -a SOURCE_DIRS=()

if [[ -n "${CONDITION}" ]]; then
  src_dir="${SRC_ROOT}/${CONDITION}/poscar"
  if [[ ! -d "${src_dir}" ]]; then
    echo "ERROR: SMOACS poscar directory not found: ${src_dir}" >&2
    exit 1
  fi
  CONDITIONS+=("${CONDITION}")
  SOURCE_DIRS+=("${src_dir}")
elif [[ -d "${SRC_ROOT}/poscar" ]]; then
  CONDITIONS+=("$(basename "${SRC_ROOT}")")
  SOURCE_DIRS+=("${SRC_ROOT}/poscar")
else
  while IFS= read -r src_dir; do
    condition="$(basename "$(dirname "${src_dir}")")"
    CONDITIONS+=("${condition}")
    SOURCE_DIRS+=("${src_dir}")
  done < <(find "${SRC_ROOT}" -mindepth 2 -maxdepth 2 -type d -name poscar | sort)
fi

if [[ "${#CONDITIONS[@]}" -eq 0 ]]; then
  echo "ERROR: no SMOACS result directories with poscar/ found in ${SRC_ROOT}" >&2
  exit 1
fi

mkdir -p "${DEST_ROOT}"

for idx in "${!CONDITIONS[@]}"; do
  condition="${CONDITIONS[$idx]}"
  src_dir="${SOURCE_DIRS[$idx]}"
  dst_dir="${DEST_ROOT}/${condition}"

  mkdir -p "${dst_dir}"

  copied_count=0
  for src_file in "${src_dir}"/*.vasp; do
    if [[ -n "${MAX_COPIES_PER_CONDITION}" ]] && [[ "${copied_count}" -ge "${MAX_COPIES_PER_CONDITION}" ]]; then
      break
    fi
    if [[ ! -f "${src_file}" ]]; then
      continue
    fi
    stem="$(basename "${src_file}" .vasp)"
    dst_file="${dst_dir}/${stem}__${condition}.vasp"
    cp "${src_file}" "${dst_file}"
    copied_count=$((copied_count + 1))
  done

  if [[ "${copied_count}" -eq 0 ]]; then
    echo "ERROR: no .vasp files found in ${src_dir}" >&2
    exit 1
  fi

  if [[ -n "${MAX_COPIES_PER_CONDITION}" ]]; then
    echo "Copied ${copied_count} files (max ${MAX_COPIES_PER_CONDITION}) -> ${dst_dir}"
  else
    echo "Copied ${copied_count} files -> ${dst_dir}"
  fi
done

echo "Completed: ${DEST_ROOT}"
