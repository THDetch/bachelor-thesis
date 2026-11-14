cd ~/.../codebase
ARM=L1-only SEED=42 EPOCHS=50 LINEAR_DECAY_START=36 NUM_WORKERS=16 \
  python train_pix2pix_lpips_ssim_generate_checkpoints_latest.py


  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 5


  chmod +x run_finals.sh
PARALLEL=1 ./run_finals.sh          # sequential, safe
PARALLEL=2 NUM_WORKERS=6 ./run_finals.sh   # two at a time

(base) dll1801@DEGGS111PC16:~/mdpi_manuscript/code/bachelor-thesis/codebase$   nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 5
memory.used [MiB], memory.total [MiB], utilization.gpu [%]
5929 MiB, 24564 MiB, 99 %
5929 MiB, 24564 MiB, 99 %
5929 MiB, 24564 MiB, 99 %
5929 MiB, 24564 MiB, 99 %


nvidia-smi --query-gpu=power.draw,power.limit,clocks.sm,clocks.max.sm --format=csv -l 5