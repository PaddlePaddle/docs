#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_FILE="$ROOT_DIR/ci_scripts/baselines/rst_single_backticks_baseline_full.txt"

if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "Baseline file not found: $BASELINE_FILE"
  exit 1
fi

PYTHON_BIN=""
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No usable Python found (.venv/bin/python or python3)."
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

CURRENT_RAW="$TMP_DIR/current_full.txt"
CURRENT_SORTED="$TMP_DIR/current_sorted.txt"
BASELINE_SORTED="$TMP_DIR/baseline_sorted.txt"
NEW_HITS="$ROOT_DIR/ci_scripts/baselines/rst_single_backticks_new_hits.txt"
NEW_HITS_BY_FILE="$ROOT_DIR/ci_scripts/baselines/rst_single_backticks_new_hits_by_file.txt"

set +e
rg --files -g '*.rst' "$ROOT_DIR" | sed "s|^$ROOT_DIR/||" | \
  xargs -r "$PYTHON_BIN" "$ROOT_DIR/ci_scripts/check_rst_single_backticks.py" \
  > "$CURRENT_RAW" 2>&1
run_status=$?
set -e

echo "Checker exit code: $run_status"

touch "$CURRENT_RAW"
sort "$CURRENT_RAW" > "$CURRENT_SORTED"
sort "$BASELINE_FILE" > "$BASELINE_SORTED"

comm -13 "$BASELINE_SORTED" "$CURRENT_SORTED" > "$NEW_HITS"

if [[ -s "$NEW_HITS" ]]; then
  awk -F: '{print $1}' "$NEW_HITS" | sort | uniq -c | sort -nr > "$NEW_HITS_BY_FILE"
else
  : > "$NEW_HITS_BY_FILE"
fi

new_total=$(wc -l < "$NEW_HITS")
new_files=$(wc -l < "$NEW_HITS_BY_FILE")

echo "New hits since baseline: $new_total"
echo "Affected files (new hits): $new_files"
echo "New hit details: $NEW_HITS"
echo "New hit summary by file: $NEW_HITS_BY_FILE"
