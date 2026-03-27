# JSSP Bucketed Batching 改造最终总结

## 交付状态

| Phase | 状态 | 测试覆盖 |
|------|------|---------|
| Phase 1: 审计 | ✅ 完成 | N/A |
| Phase 2: 10x10 Same-Shape RL | ✅ 完成 | 6/6 测试 |
| Phase 3: 10x10 Same-Shape PO/BOPO | ✅ 完成 | 4/4 测试 |
| Phase 4: Bucket-by-Shape (10/15/20) | ✅ 完成 | 3/3 测试 |
| Phase 5: 全链路 Bucket-by-Shape | ✅ 完成 | 11/11 测试 |
| Phase 6: 全验证 | ✅ 完成 | 13/13 测试 |
| Phase 7: Eval 多 Batch + 最终总结 | ✅ 完成 | 2/2 测试 |

**总计测试：39/39 全部通过** ✅

---

## 1. 当前已经支持的 Shape

| Shape | 状态 | 说明 |
|-------|------|------|
| 10x10 | ✅ 完全支持 | Phase 2 correctness milestone |
| 15x15 | ✅ 完全支持 | Phase 4 扩展 |
| 20x20 | ✅ 完全支持 | Phase 4 扩展 |
| 任意 same-shape | ✅ 支持 | 代码通用，不限于上述三个 |

## 2. RL / PO / BOPO 支持范围

| 路径 | Train Same-Shape | Train Bucket-by-Shape | Val/Test Multi-Batch | 实例内 Pair |
|-----|------------------|---------------------|---------------------|-----------|
| **RL** | ✅ batch_size>1 | ✅ | ✅ | N/A |
| **PO** | ✅ batch_size>1 | ✅ | ✅ | ✅ |
| **BOPO** | ✅ batch_size>1 | ✅ | ✅ | ✅ |

### 关键张量形状

设 `N = batch_size（实例数）`，`B = 每实例 rollout 数`，`J = num_jobs`，`M = num_machines`，`S = J*M-1`

| Tensor | RL/PO/BOPO 形状 |
|--------|-----------------|
| solve_jsp trajs | `(N*B, S)` |
| solve_jsp logits | `(N*B, S, J)` |
| solve_jsp makespans | `(N*B,)` |
| (reshape) 单实例 trajs | `(B, S)` |
| (reshape) 单实例 logits | `(B, S, J)` |
| 最终 loss | scalar, `avg(Li for i in 1..N)` |

## 3. 训练配置

### 方案 A: Single-Shape Only (传统模式)
```yaml
model:
  baseline: rl  # 或 po / bopo
  batch_size: 4
  use_shape_buckets: false  # 禁用
```

### 方案 B: Multi-Shape Bucketed (推荐)
```yaml
model:
  baseline: rl  # 或 po / bopo
  batch_size: 4  # per-shape batch size
  use_shape_buckets: true
  bucket_drop_last: false
  allowed_shapes: [[10, 10], [15, 15], [20, 20]]  # 可选，不过滤则留空
  log_on_step: true  # 可选，查看当前 batch shape
```

### 已提供的配置文件
- `configs/experiment/scheduling/mgl-jssp-rl-bucketed-multishape.yaml`
- `configs/experiment/scheduling/mgl-jssp-po-bucketed-multishape.yaml`
- `configs/experiment/scheduling/mgl-jssp-bopo-bucketed-multishape.yaml`

### 预期日志特征

当 `log_on_step: true` 时，训练日志会显示：
```
train/shape_j: 10.0  (当前 batch 是 10x10)
train/shape_m: 10.0
...
train/shape_j: 15.0  (下一个 batch 是 15x15)
train/shape_m: 15.0
```

Setup 阶段会打印 bucket stats：
```
=== JSSP Shape Bucket Stats ===
  Shape 10x10: 128 instances, 32 batches
  Shape 15x15: 96 instances, 24 batches
  Shape 20x20: 64 instances, 16 batches
  TOTAL: 288 instances, 72 batches
===============================
```

## 4. 目前还没支持的内容

| 功能 | 状态 | 说明 |
|-----|------|------|
| **True Mixed-Shape Padding** | ❌ | 仅 bucket sampler，无 padding |
| **L2D 模型** | ❌ | 仅 MGL 模型改造 |
| **PyG Batch 优化** | ❌ | GAT 仍用 for 循环 |
| **Distributed Training** | ❌ | 未测试多卡 |
| **FFSP / FJSP** | ❌ | 仅 JSSP |

## 5. 如果以后要做 True Mixed-Shape Padding

需要修改的地方：

| 模块 | 修改内容 | 难度 |
|-----|---------|------|
| `data.py:collate_fn` | 不再拒绝 mixed-shape，改为 padding 成 max shape | 中 |
| `sampling.py:JobShopStates` | 支持 padded ops/jobs/machines，加 `pad_mask` | 高 |
| `sampling.py:solve_jsp` | 处理 variable `num_jobs` / `num_machines` | 高 |
| `sampling.py:sample_training_pair` | Pair 构造时忽略 padding 区域 | 中 |
| `sampling.py:po_loss` / `sro_loss` | Loss 计算时 mask padding | 中 |
| `model.py` | 配置选项 `use_padding: true/false` | 低 |

**关键设计决策**：
- 选择 padding 到 batch 内 max shape，还是全局 max shape
- `traj` / `logits` 如何处理 variable `num_steps` (J*M-1)

## 6. 如果以后要扩到 FJSP / FFSP

最可能复用的模块：

| 模块 | FJSP 复用性 | FFSP 复用性 | 说明 |
|-----|------------|------------|------|
| `JSSPShapeBucketSampler` | ✅ 高 | ✅ 高 | 只要 dataset 有 `get_shape(idx)` 即可 |
| `JSSPInstanceDataset.collate_fn` | ✅ 高 | ✅ 高 | Same-shape 检查通用 |
| `model.py:train_dataloader` | ✅ 高 | ✅ 高 | Bucket sampler 逻辑通用 |
| `model.py:setup` bucket stats | ✅ 高 | ✅ 高 | 仅需修改 shape 提取 |
| `model.py:training_step` shape logs | ✅ 高 | ✅ 高 | 通用 |
| `model.py:_training_rollout` loss 聚合 | ✅ 高 | ✅ 高 | Per-instance 聚合通用 |
| `sampling.py:JobShopStates` | ⚠️ 中 | ❌ 低 | FJSP/FFSP 状态空间不同 |
| `sampling.py:solve_jsp` | ⚠️ 中 | ❌ 低 | 环境 rollout 不同 |

**推荐复用策略**：
1. **Data 层**：100% 复用 `JSSPShapeBucketSampler`，改名 `ShapeBucketSampler`
2. **Model 层**：100% 复用 bucket sampler 接入、loss 聚合、shape logging
3. **Sampling 层**：重写环境特定的 rollout，但保留 per-instance loss 聚合模式

---

## 测试命令

```bash
# 运行所有 JSSP 测试
conda run -n rlco python -m pytest tests/test_phase2_batched_rl.py tests/test_phase3_batched_po_bopo.py tests/test_phase4_bucketed_batching.py tests/test_phase5_all_paths_bucketed.py tests/test_phase6_full_validation.py tests/test_phase7_eval_multi_batch.py tests/test_mgl_jssp_data.py tests/test_jssp_mgl_losses.py -v
```

## 修改文件清单

| 文件 | 类型 |
|------|------|
| `rl4co/models/zoo/mgl_jssp/data.py` | 修改 |
| `rl4co/models/zoo/mgl_jssp/sampling.py` | 修改 |
| `rl4co/models/zoo/mgl_jssp/model.py` | 修改 |
| `configs/experiment/scheduling/mgl-jssp-rl-batch-10x10.yaml` | 新增 |
| `configs/experiment/scheduling/mgl-jssp-po-batch-10x10.yaml` | 新增 |
| `configs/experiment/scheduling/mgl-jssp-bopo-batch-10x10.yaml` | 新增 |
| `configs/experiment/scheduling/mgl-jssp-rl-bucketed-multishape.yaml` | 新增 |
| `configs/experiment/scheduling/mgl-jssp-po-bucketed-multishape.yaml` | 新增 |
| `configs/experiment/scheduling/mgl-jssp-bopo-bucketed-multishape.yaml` | 新增 |
| `tests/test_phase2_batched_rl.py` | 新增 |
| `tests/test_phase3_batched_po_bopo.py` | 新增 |
| `tests/test_phase4_bucketed_batching.py` | 新增 |
| `tests/test_phase5_all_paths_bucketed.py` | 新增 |
| `tests/test_phase6_full_validation.py` | 新增 |
| `tests/test_phase7_eval_multi_batch.py` | 新增 |
| `docs/JSSP_BUCKETED_BATCHING_SUMMARY.md` | 新增（本文档）|

---

**Phase 1-7 全部完成！** 🎉
