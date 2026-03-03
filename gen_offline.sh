python scripts/eval_baseline_minitrain.py \
  --config PTP/configs/experiment/pref_loss_coevo/alternating_simple.yaml \
  --K 200 \
  --train_problem_size 100 \
  --valid_problem_sizes 100 \
  --num_validation_episodes 512 \
  --train_batch_size 64 \
  --offline_train offline_data/tsp100_train.pt \
  --offline_val 100 offline_data/tsp100_val.pt \
  --scratch_init_seed 12345 \
  --ckpt_135 baseline/epoch_135.ckpt \
  --ckpt_409 baseline/epoch_409.ckpt \
  --out baseline/mini_eval/baseline_minitrain_tsp100_K200.json

python scripts/eval_baseline_minitrain.py \
  --config PTP/configs/experiment/pref_loss_coevo/alternating_simple.yaml \
  --K 1000 \
  --train_problem_size 100 \
  --valid_problem_sizes 100 \
  --num_validation_episodes 512 \
  --train_batch_size 64 \
  --offline_train offline_data/tsp100_train.pt \
  --offline_val 100 offline_data/tsp100_val.pt \
  --scratch_init_seed 12345 \
  --ckpt_135 baseline/epoch_135.ckpt \
  --ckpt_409 baseline/epoch_409.ckpt \
  --out baseline/mini_eval/baseline_minitrain_tsp100_K1000.json