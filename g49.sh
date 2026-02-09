# nohup python run_kp_eam_parallel.py --cuda 0 --epochs 10 --log-dir logs > kp.log 2>&1 &
# nohup python run_epoch_timing.py --cuda 1 --output results/epoch_timing_sampled3.csv --profile-steps 50 --profile-warmup 5 > timing.log 2>&1 &
nohup python run_epoch_timing.py \
    --suite tevc \
    --cuda 0 \
    --output results/epoch_timing_tevc.csv \
    --profile-steps 500 \
    --profile-warmup 0 \
    --exclude-problems kp > timing_tevc.log 2>&1 &

# Non-EAM baselines (AM, POMO, SymNCO) on the full TEVC problem set.
# Same sampling setup as epoch_timing_tevc.csv (profile-steps=500, warmup=0).
nohup python run_epoch_timing.py \
    --suite tevc_base \
    --cuda 0 \
    --output results/epoch_timing_tevc_base.csv \
    --profile-steps 500 \
    --profile-warmup 0 > timing_tevc_base.log 2>&1 &

# python run_gaussian_rosenblatt_eval.py --ckpt-path checkpoints/pomo_tsp100.ckpt --num-instances 1000 --seed 0 --device cuda
