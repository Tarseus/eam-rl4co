export cuda_visible_devices=0
export TZ=Asia/Shanghai

nohup python -u eval_checkpoints.py \
    --method multistart_greedy \
    --seeds 0-9 \
    --num-instances 1000 > multistart.log 2>&1 &

# nohup python -u eval_checkpoints.py \
#     --method multistart_greedy_augment \
#     --num-augment 8 \
#     --seeds 0-9 \
#     --num-instances 1000 > multistart_augment.log 2>&1 &

# python -u eval_checkpoints.py \
#     --method multistart_greedy \
#     --seeds 0-9 \
#     --num-instances 1000

# python -u eval_checkpoints.py \
#     --method multistart_greedy_augment \
#     --num-augment 8 \
#     --seeds 0-9 \
#     --num-instances 1000