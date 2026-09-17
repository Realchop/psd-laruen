#!/usr/bin/env bash
# One cell of the comparison grid.
#
#   sbatch scripts/train.sbatch <amp> <model> <config> [learning_rate]
#
# e.g. sbatch scripts/train.sbatch preamp AmpFormer configs/ampformer_48ms.json
#
# A fourth argument overrides the config's learning rate and tags the run with
# it, which is how scripts/lr_sweep.sh varies one hyperparameter at a time.
#
# Runs on the msp3 nodes: every other zone is at a different site and cannot
# see this filesystem. Pin by zone, not by gres -- sof1 has h200s too.
#SBATCH --job-name=amp
#SBATCH --partition=batch
#SBATCH --constraint=zone-msp3
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

AMP="${1:?usage: train.sbatch <amp> <model> <config>}"
MODEL="${2:?usage: train.sbatch <amp> <model> <config>}"
CONFIG="${3:?usage: train.sbatch <amp> <model> <config>}"
LR="${4:-}"

PROJECT=/home/vasilije_ivanovic/MlAmp/psd-laruen
cd "$PROJECT"

# Off by default: on this model the range test reports a rate orders of
# magnitude below the one that trains, because the loss rises off the
# near-passthrough initialisation before it falls. FIND_LR=1 runs it anyway.
FIND_LR_FLAG=()
if [[ "${FIND_LR:-0}" == "1" ]]; then
    FIND_LR_FLAG=(--find-lr)
fi

# An explicit rate is a sweep point, so keep it out of the baseline's logs.
LR_FLAG=()
TAG="${AMP}-$(basename "$CONFIG" .json)"
if [[ -n "$LR" ]]; then
    LR_FLAG=(--learning-rate "$LR")
    TAG="${TAG}-lr${LR}"
fi
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
    --name "$TAG" \
    ${LR_FLAG[@]+"${LR_FLAG[@]}"} \
    ${FIND_LR_FLAG[@]+"${FIND_LR_FLAG[@]}"}
