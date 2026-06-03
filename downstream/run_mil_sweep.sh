#!/usr/bin/env bash
set -e

SLIDES_CSV=$1
FEATURE_DIR=$2
OUT_ROOT=$3
DEVICE=${4:-cuda}

if [ -z "$SLIDES_CSV" ] || [ -z "$FEATURE_DIR" ] || [ -z "$OUT_ROOT" ]; then
  echo "Usage: bash downstream/run_mil_sweep.sh <slides_csv> <feature_dir> <out_root> [device]"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

ATT_DIMS=(128 256)
LRS=(1e-4 5e-5)
WDS=(1e-3 1e-4)
MAX_INSTS=(None)

for ATT_DIM in "${ATT_DIMS[@]}"; do
  for LR in "${LRS[@]}"; do
    for WD in "${WDS[@]}"; do
      for MAX_INST in "${MAX_INSTS[@]}"; do

        EXP_NAME="abmil_att${ATT_DIM}_lr${LR}_wd${WD}_mi${MAX_INST}"
        OUT_DIR="${OUT_ROOT}/${EXP_NAME}"

        echo "======================================================"
        echo "[RUN] ${EXP_NAME}"
        echo "======================================================"

        python downstream/train_abmil.py \
          --slides_csv "${SLIDES_CSV}" \
          --feature_dir "${FEATURE_DIR}" \
          --out_dir "${OUT_DIR}" \
          --device "${DEVICE}" \
          --mil_model abmil \
          --att_dim "${ATT_DIM}" \
          --epochs 20 \
          --patience 5 \
          --lr "${LR}" \
          --weight_decay "${WD}" \
          --max_instances "${MAX_INST}" \
          --shuffle_instances \
          --monitor auc \
          --batch_size 1 \
          --num_workers 4 \
          --seed 42

      done
    done
  done
done