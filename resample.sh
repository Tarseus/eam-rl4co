# nohup python run_eam_tsp100_resample.py --cuda 2 > resample.log 2>&1 &

nohup python run_eam_tsp100_random2opt.py --cuda 4 > random.log 2>&1 &

nohup python run_eam_tsp100_localsearch.py --cuda 5 > localsearch.log 2>&1 &

# nohup python run_cvrp100_pomo_eam.py --model eam-pomo --epochs 200 --seed 0 --device 5 --log-dir logs --run-name eam-pomo_cvrp100_seed0 > eam_pomo_cvrp100_seed0.log 2>&1 &