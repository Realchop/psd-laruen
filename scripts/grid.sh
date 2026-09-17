#!/usr/bin/env bash
# The main comparison grid: 4 amps x 2 architectures x 2 context lengths.
#
#   ./scripts/grid.sh
#
# Every cell trains at the rate its config pins, so the only thing that varies
# across the grid is the architecture and its context. Varying the rate is
# scripts/lr_sweep.sh, kept separate so the two cannot confound each other.
set -euo pipefail

cd "$(dirname "$0")/.."

AMPS=(clean cabinet preamp metal)
CELLS=(
    "WaveNet configs/wavenet_46ms.json"
    "WaveNet configs/wavenet_186ms.json"
    "AmpFormer configs/ampformer_48ms.json"
    "AmpFormer configs/ampformer_187ms.json"
)

for amp in "${AMPS[@]}"; do
    for cell in "${CELLS[@]}"; do
        read -r model config <<<"$cell"
        printf '%-8s %-10s %-32s -> ' "$amp" "$model" "$config"
        sbatch scripts/train.sbatch "$amp" "$model" "$config"
    done
done
