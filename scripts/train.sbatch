#!/usr/bin/env bash
# One cell of the comparison grid.
#
#   sbatch scripts/train.sbatch <amp> <model> <config>
#
# e.g. sbatch scripts/train.sbatch preamp AmpFormer configs/ampformer_48ms.json
#
# Runs on the msp3 nodes: `hala` (the a6000 partition) is at a different site
# and cannot see this filesystem.
#SBATCH --job-name=amp
#SBATCH --partition=batch
#SBATCH --exclude=hala
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

AMP="${1:?usage: train.sbatch <amp> <model> <config>}"
MODEL="${2:?usage: train.sbatch <amp> <model> <config>}"
CONFIG="${3:?usage: train.sbatch <amp> <model> <config>}"

PROJECT=/home/vasilije_ivanovic/MlAmp/psd-laruen
cd "$PROJECT"

TAG="${AMP}-$(basename "$CONFIG" .json)"
echo "amp=$AMP model=$MODEL config=$CONFIG tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Train on the Ibanez, validate on the Career SG: a different guitar is a real
# generalisation test, where a random split of one recording is not.
exec .venv/bin/train \
    "data/Ibanez2820-DI.wav" \
    "data/targets/Ibanez2820-DI__${AMP}.wav" \
    --val-dry "data/Carrer SG.wav" \
    --val-wet "data/targets/CareerSG__${AMP}.wav" \
    --model "$MODEL" \
    --config "$CONFIG" \
    --epochs "${EPOCHS:-20}" \
    --batch "${BATCH:-16}" \
    --workers 8 \
    --name "$TAG"
