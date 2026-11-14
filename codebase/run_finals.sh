#!/usr/bin/env bash
# Final runs: {L1-only, A3, A3-g3} x {42, 43, 44}, N=50, linear LR decay from epoch 36.
#
#   ./run_finals.sh            # defaults below
#   PARALLEL=2 ./run_finals.sh # two at a time
#   PARALLEL=1 ./run_finals.sh # strictly sequential
#
# Order pairs different arms together, so a crash costs one seed of each configuration
# rather than one whole configuration.

set -u

SCRIPT=train_pix2pix_lpips_ssim_generate_checkpoints_latest.py
EPOCHS=${EPOCHS:-50}
LINEAR_DECAY_START=${LINEAR_DECAY_START:-36}
PARALLEL=${PARALLEL:-1}
NUM_WORKERS=${NUM_WORKERS:-6}
GPU_ID=${GPU_ID:-0}
LOGDIR=${LOGDIR:-logs_finals}

JOBS=(
  "L1-only 42"  "A3-g3 42"
  "A3 42"       "L1-only 43"
  "A3-g3 43"    "A3 43"
  "L1-only 44"  "A3-g3 44"
  "A3 44"
)

mkdir -p "$LOGDIR"

run_one() {
  local arm=$1 seed=$2
  local out="trained_arm_${arm}_s${seed}"
  if [ -f "${out}/pix2pix/history.csv" ] && \
     [ "$(wc -l < "${out}/pix2pix/history.csv")" -gt "$EPOCHS" ]; then
    echo "SKIP  ${arm} seed ${seed}: ${out} already has ${EPOCHS} epochs"
    return 0
  fi
  echo "START ${arm} seed ${seed} -> ${out}  (log ${LOGDIR}/${arm}_s${seed}.log)"
  ARM="$arm" SEED="$seed" EPOCHS="$EPOCHS" \
  LINEAR_DECAY_START="$LINEAR_DECAY_START" \
  NUM_WORKERS="$NUM_WORKERS" GPU_ID="$GPU_ID" \
    python "$SCRIPT" > "${LOGDIR}/${arm}_s${seed}.log" 2>&1
  local rc=$?
  echo "DONE  ${arm} seed ${seed}  exit ${rc}"
  return $rc
}

echo "epochs=${EPOCHS} decay_from=${LINEAR_DECAY_START} parallel=${PARALLEL} workers=${NUM_WORKERS}"
echo

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for _ in $(seq 1 "$PARALLEL"); do
    [ $i -ge ${#JOBS[@]} ] && break
    read -r arm seed <<< "${JOBS[$i]}"
    run_one "$arm" "$seed" &
    i=$((i + 1))
    sleep 20   # stagger so two processes do not allocate CUDA at the same instant
  done
  wait
done

echo
echo "all finished. check for aborts:"
echo "  grep -l ABORT ${LOGDIR}/*.log"
