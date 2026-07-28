# RFPS 最小充分描述符消融协议

## 研究问题

在固定真实候选、真实 rollout probe 和真实训练适应度标签时，当前三点
Fisher--Rao 曲率残差中哪些组件具有不可替代的增量价值？目标不是继续提高
方法复杂度，而是在保持筛选有效性的前提下删去冗余机制。

## 数据与冻结项

- 匹配双通道集合：40 个候选，同时具有 from-scratch 与 checkpoint-135
  高保真适应度。
- 外部复现集合：65 个未参与曲率公式选择的 scratch 高保真候选。
- 每个锚点使用两组独立 rollout bank；每组 8 个 TSP100 实例、每实例
  100 个 POMO starts。
- 所有描述符在同一候选、同一 probe、同一真实适应度标签上比较。
- 主 Fisher 流使用共同弧长 0.10、30 个积分步和 ESS 下界 0.4。

## 预注册描述符

1. `initial_field`：共同起点上的完整 Fisher--Rao 切向量，而非梯度范数。
2. `raw_three_positions`：三个流位置直接拼接。
3. `kappa_full`：当前两个相邻归一化位移差的拼接。
4. `kappa_early`：只保留前两个流点形成的第一个归一化位移差；检验第三个
   流点是否冗余。
5. `kappa_late`：只保留后两个流点形成的第二个归一化位移差。
6. `field_delta_fisher`：沿 Fisher 流前进一个检查点后，比较相邻两点的
   完整梯度场；它是廉价的两点梯度差 / 有限差分 JVP 基线。
7. `euclidean_field_delta`：在固定有效 logits 上做等长度欧氏虚拟更新后，
   比较两点梯度场；检验 Fisher 几何是否必要。
8. `euclidean_curvature`：固定 logits 上三点欧氏轨迹的二阶残差。
9. `initial_gram`：只用各 probe 初始完整梯度构造候选内 Gram 描述符。
10. `random_projection`：对完整曲率做固定随机投影，检验高维坐标是否必要。

## 评价指标

令描述符距离为 `d_ij`，真实训练适应度差为
`y_ij = |F_i - F_j|`。主指标为：

1. `rho`：`d_ij` 与 `y_ij` 的 Spearman 相关，越大越好；
2. `NN error`：描述符最近邻的真实适应度绝对误差中位数，越小越好；
3. `false-skip`：按历史顺序、在固定跳过分位数和适应度容差下的误跳过率；
4. `cost`：每个候选需要的流点数、重新求导次数和描述符维数。

## 预注册删减规则

把当前 `kappa_full` 作为参照。更简单描述符只有同时满足以下条件才可替代：

1. 在两组 matched-scratch probe 和两组 external-scratch probe 上，
   `rho >= rho_full - 0.05`；
2. 在上述四个 scratch 设置上，
   `NN error <= 1.10 * NN_error_full`；
3. 在两组 warmstart probe 上不出现超过 0.10 的 `rho` 下降；
4. 在 20% 顺序跳过率、适应度容差 0.01 下，误跳过率不比完整曲率增加
   5 个百分点以上；
5. 至少减少一个实质组件：流检查点、重新求导次数、几何运算或 probe 数。

若没有单一简化描述符满足全部规则，则保留完整曲率，但仍删除实验证明无效的
辅助组件。所有未预注册的组合或阈值均标为探索性结果。

## 假设

- H1：`kappa_early` 足以匹配 `kappa_full`，第三个流点可以删除。
- H2：`field_delta_fisher` 若匹配完整曲率，则对数映射和三点二阶残差可以
  改写为更简单的两点梯度变化。
- H3：若欧氏基线匹配完整曲率，则 Fisher--Rao 几何不是必要组件。
- H4：若少量 probe 已稳定，则多探针拼接可以缩减为更小固定 probe bank。
- H5：若跨锚点描述符不损失预测力，则 scratch/warm 双锚点可以合并。

## 解释边界

当前候选仍为固定 `g_ref`、变化 `f`。本实验决定现有描述符的最小形式，不
外推为完整 `(f,g)` 联合搜索的最终性能证明，也不把离线相关直接解释为
best-so-far 改善。
