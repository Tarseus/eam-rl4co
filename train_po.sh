nohup python -u run.py experiment=routing/pomo-po4cops-tsp100-po \
  ~callbacks.learning_rate_monitor ~callbacks.rich_progress_bar \
  > train_po.log 2>&1 &