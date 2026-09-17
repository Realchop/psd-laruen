#!/usr/bin/env bash
# The learning rate sweep: one hyperparameter varied, everything else fixed.
#
#   ./scripts/lr_sweep.sh [amp]
#
# Both architectures are swept over the same rates on the same amp, at their
# short-context configs, so the comparison is between rates and not between
# run lengths. The baseline rate the main grid trains at (1e-3) is included as
# a sweep point, so the curve does not have to be read across two experiments.
set -euo pipefail

cd "$(dirname "$0")/.."

AMP="${1:-metal}"
RATES=(1e-4 3e-4 1e-3 3e-3 1e-2)
CELLS=(
    "WaveNet configs/wavenet_46ms.json"
    "AmpFormer configs/ampformer_48ms.json"
)

for cell in "${CELLS[@]}"; do
    read -r model config <<<"$cell"
    for lr in "${RATES[@]}"; do
        printf '%-10s %-32s lr=%-6s -> ' "$model" "$config" "$lr"
        sbatch scripts/train.sbatch "$AMP" "$model" "$config" "$lr"
    done
done
