#!/usr/bin/env bash
# Final runs: {L1-only, A3, A3-g3} x {42, 43, 44}, N=50, linear LR decay from epoch 36.
#
#   ./run_finals.sh            # sequential, live output + logs
#   PARALLEL=2 ./run_finals.sh # two at a time (only if the GPU is not saturated)
#
# Order pairs different arms together, so a crash costs one seed of each configuration
# rather than one whole configuration.

set -u

SCRIPT=train_pix2pix_lpips_ssim_generate_checkpoints_latest.py
CONDA_ENV=${CONDA_ENV:-pytorch-img2img}
EPOCHS=${EPOCHS:-50}
LINEAR_DECAY_START=${LINEAR_DECAY_START:-36}
PARALLEL=${PARALLEL:-1}
NUM_WORKERS=${NUM_WORKERS:-16}
GPU_ID=${GPU_ID:-0}
LOGDIR=${LOGDIR:-logs_finals}

JOBS=(
  "L1-only 42"  "A3-g3 42"
  "A3 42"       "L1-only 43"
  "A3-g3 43"    "A3 43"
  "L1-only 44"  "A3-g3 44"
  "A3 44"
)

# ---- environment -------------------------------------------------------------
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  if [ -z "${CONDA_EXE:-}" ]; then
    echo "ERROR: conda not on PATH and CONDA_EXE unset; activate $CONDA_ENV manually" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate $CONDA_ENV" >&2; exit 1; }
fi

echo "env     : ${CONDA_DEFAULT_ENV:-?}  ($(which python))"
python - <<'PY' || exit 1
import sys, torch
print(f"torch   : {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA not available")
PY
echo "config  : epochs=${EPOCHS} decay_from=${LINEAR_DECAY_START} parallel=${PARALLEL} workers=${NUM_WORKERS}"
echo

mkdir -p "$LOGDIR"

run_one() {
  local arm=$1 seed=$2
  local out="trained_arm_${arm}_s${seed}"
  local log="${LOGDIR}/${arm}_s${seed}.log"

  if [ -f "${out}/pix2pix/history.csv" ] && \
     [ "$(wc -l < "${out}/pix2pix/history.csv")" -gt "$EPOCHS" ]; then
    echo "SKIP  ${arm} seed ${seed}: ${out} already has ${EPOCHS} epochs"
    return 0
  fi

  echo "========================================================================"
  echo "START ${arm} seed ${seed} -> ${out}   log: ${log}"
  echo "========================================================================"

  # -u keeps print() unbuffered through the pipe; tee gives live output and a log.
  ARM="$arm" SEED="$seed" EPOCHS="$EPOCHS" \
  LINEAR_DECAY_START="$LINEAR_DECAY_START" \
  NUM_WORKERS="$NUM_WORKERS" GPU_ID="$GPU_ID" \
    python -u "$SCRIPT" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}

  echo "DONE  ${arm} seed ${seed}  exit ${rc}"
  echo
  return "$rc"
}

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  if [ "$PARALLEL" -le 1 ]; then
    read -r arm seed <<< "${JOBS[$i]}"
    run_one "$arm" "$seed"
    i=$((i + 1))
  else
    for _ in $(seq 1 "$PARALLEL"); do
      [ $i -ge ${#JOBS[@]} ] && break
      read -r arm seed <<< "${JOBS[$i]}"
      # Parallel output would interleave unreadably, so log only.
      ( run_one "$arm" "$seed" > /dev/null ) &
      i=$((i + 1))
      sleep 20
    done
    wait
  fi
done

echo "all finished. check for aborts:"
echo "  grep -l ABORT ${LOGDIR}/*.log"
