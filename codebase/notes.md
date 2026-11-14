cd ~/.../codebase
ARM=L1-only SEED=42 EPOCHS=50 LINEAR_DECAY_START=36 NUM_WORKERS=16 \
  python train_pix2pix_lpips_ssim_generate_checkpoints_latest.py


  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 5


  chmod +x run_finals.sh
PARALLEL=1 ./run_finals.sh          # sequential, safe
PARALLEL=2 NUM_WORKERS=6 ./run_finals.sh   # two at a time

