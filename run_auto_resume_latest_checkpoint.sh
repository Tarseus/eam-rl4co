: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
nohup python PTP/ptp_discovery/run_pref_loss_coevo.py \
  --config PTP/configs/experiment/pref_loss_coevo/alternating_simple.yaml \
  --resume-latest \
  > logs/pref_loss_alternating_simple_resume_$(date +%Y%m%d-%H%M%S).out 2>&1 &

tail -f logs/pref_loss_alternating_simple_resume_$(date +%Y%m%d-%H%M%S).out