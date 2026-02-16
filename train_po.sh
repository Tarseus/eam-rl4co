nohup python run.py experiment=routing/pomo logger=none \
  env.generator_params.num_loc=100 \
  model.loss_type=po_loss model.alpha=0.05 \
  +model.num_starts=100 \
  model.batch_size=64 model.train_data_size=100000 \
  +model.optimizer=Adam model.optimizer_kwargs.lr=3e-4 model.optimizer_kwargs.weight_decay=1e-6 \
  model.lr_scheduler=MultiStepLR model.lr_scheduler_kwargs.milestones=[3001] model.lr_scheduler_kwargs.gamma=0.2 \
  trainer.max_epochs=2010 +trainer.accumulate_grad_batches=1 \
  trainer.accelerator=gpu \
  +trainer.devices=[1] \
  +trainer.strategy=auto trainer.precision=32-true\
  +model.policy_kwargs.embed_dim=128 \
  +model.policy_kwargs.num_encoder_layers=6 \
  +model.policy_kwargs.num_heads=8 \
  +model.policy_kwargs.feedforward_hidden=512 \
  +model.policy_kwargs.tanh_clipping=50 \
  +model.policy_kwargs.val_decode_type=greedy \
  logger=csv \
  ~callbacks.learning_rate_monitor \
  ~callbacks.rich_progress_bar \
  +trainer.enable_progress_bar=false \
  +trainer.log_every_n_steps=50 \
  +model.policy_kwargs.test_decode_type=greedy > train_po.log 2>&1 &