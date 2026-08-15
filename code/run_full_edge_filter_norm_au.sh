#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROFILE_FILE="${PROFILE_FILE:-$script_dir/../configs/full_au_edge_filter_norm.json}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v5.5_au_full_noise_curve_lr_split}"

exec bash "$script_dir/run_full_edge_filter_norm_ssm.sh"
