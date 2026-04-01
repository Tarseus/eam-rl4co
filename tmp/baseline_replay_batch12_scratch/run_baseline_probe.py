import json
import sys
from pathlib import Path
repo = Path(r"E:\CAS\rl4co-repo\eam-rl4co")
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / 'PTP'))
import torch
import yaml
from fitness.free_loss_fidelity import evaluate_po_baseline_rl4co
from ptp_discovery.pref_loss_coevo_loop import _build_hf_cfg, _resolve_training_seed

torch.cuda.set_device('cuda:0')
cfg = yaml.safe_load((repo / 'tmp' / 'local_ffsp100_batch12_single_gpu.yaml').read_text(encoding='utf-8'))
cfg['train_batch_size'] = 12
cfg['validation_batch_size'] = 12
cfg['hf_epochs'] = 10
cfg['scratch_hf_epochs'] = 10
cfg['warmstart_hf_epochs'] = 0
seed = int(_resolve_training_seed(cfg))
hf_cfg = _build_hf_cfg(cfg, seed=seed, device_str='cuda:0')
fit = evaluate_po_baseline_rl4co(
    hf_cfg,
    init_checkpoint_path=None,
    init_checkpoint_epoch=None,
    scratch_hf_epochs=10,
    warmstart_hf_epochs=0,
    baseline_epoch_compare_offset=int(cfg.get('baseline_epoch_compare_offset', 0) or 0),
    baseline_epoch_violation_weight=float(cfg.get('baseline_epoch_violation_weight', 1.0)),
    baseline_epoch_tail_frac=float(cfg.get('baseline_epoch_tail_frac', 1.0) or 1.0),
    baseline_epoch_window_k=int(cfg.get('baseline_epoch_window_k', 10) or 10),
    baseline_epoch_window_violation_weight=float(cfg.get('baseline_epoch_window_violation_weight', 0.0) or 0.0),
)
print(json.dumps({'validation_objective': fit.get('validation_objective'), 'size_objectives': fit.get('size_objectives')}, ensure_ascii=False))
