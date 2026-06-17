#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY_PATH="${BINARY_PATH:-$SCRIPT_DIR/../build/ismMergeTreeSimplifier_standalone}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/../dataset_gyc/gyc}"
NODES_PATH="${1:-$DATA_DIR/tangaroa_port_nodes.vtu}"
ARCS_PATH="${2:-$DATA_DIR/tangaroa_port_arcs.vtu}"
SEG_PATH="${3:-$DATA_DIR/tangaroa_port_2.vti}"

if [[ ! -x "$BINARY_PATH" ]]; then
  echo "Binary not found or not executable: $BINARY_PATH" >&2
  exit 1
fi

for input_path in "$NODES_PATH" "$ARCS_PATH" "$SEG_PATH"; do
  if [[ ! -f "$input_path" ]]; then
    echo "Input file not found: $input_path" >&2
    exit 1
  fi
done

timestamp="$(date +%Y%m%d_%H%M%S)"
result_root="$SCRIPT_DIR/result/wide_thresholds_${timestamp}"
summary_path="$result_root/summary.txt"

mkdir -p "$result_root"

configs=(
  "very_strict|0.02|0.05|0.02|Very conservative"
  "strict|0.25|1.8|0.18|Conservative"
  "loose|0.35|2.5|0.25|Aggressive"
  "very_loose|0.60|5.0|0.50|Very aggressive"
)

{
  echo "ISM Merge Tree Simplifier - Wide Threshold Sweep"
  echo "================================================"
  echo
  echo "Output directory: $result_root"
  echo "Nodes: $NODES_PATH"
  echo "Arcs: $ARCS_PATH"
  echo "Segmentation: $SEG_PATH"
  echo
} > "$summary_path"

for config in "${configs[@]}"; do
  IFS="|" read -r label epsilon_ism epsilon_pos epsilon_bbox description <<< "$config"
  run_dir="$result_root/$label"
  output_prefix="$run_dir/$label"
  stats_path="${output_prefix}_stats.csv"

  mkdir -p "$run_dir"

  echo "Running $label: ism=$epsilon_ism pos=$epsilon_pos bbox=$epsilon_bbox"
  "$BINARY_PATH" \
    "$NODES_PATH" \
    "$ARCS_PATH" \
    "$SEG_PATH" \
    -o "$output_prefix" \
    -e "$epsilon_ism" \
    -p "$epsilon_pos" \
    -b "$epsilon_bbox"

  kept_nodes="$(sed -n 's/^Total nodes kept: //p' "$stats_path")"
  if [[ -z "$kept_nodes" ]]; then
    kept_nodes="unknown"
  fi

  {
    echo "$label"
    echo "  Description: $description"
    echo "  epsilonIsm: $epsilon_ism"
    echo "  epsilonPos: $epsilon_pos"
    echo "  epsilonBbox: $epsilon_bbox"
    echo "  Total nodes kept: $kept_nodes"
    echo "  Output prefix: $output_prefix"
    echo
  } >> "$summary_path"
done

echo "Sweep complete. Summary written to $summary_path"
