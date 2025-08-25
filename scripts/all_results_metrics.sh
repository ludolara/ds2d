#!/usr/bin/env bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="results"

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "ERROR: '$RESULTS_DIR' not found at $ROOT_DIR" >&2
  exit 1
fi

shopt -s nullglob
for EXP_DIR in "$RESULTS_DIR"/*/; do
  [[ -d "$EXP_DIR" ]] || continue
  EXP_DIR_NO_SLASH="${EXP_DIR%/}"
  echo "Running scripts/metric.sh on: $EXP_DIR_NO_SLASH"
  bash scripts/metric.sh "$EXP_DIR_NO_SLASH" || echo "FAILED: $EXP_DIR_NO_SLASH" >&2
  echo
done

echo "Done." 