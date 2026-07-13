# Discussion

本文的核心发现是，preference objective 的设计可以从经验性的人工选择，转化为一个可评估的 discovery process。这个过程的价值不只体现在最终找到更好的 training objective；更重要的是，搜索后存活下来的 objectives、搜索轨迹中的改进路径、以及跨规模迁移中的失败模式，共同揭示了 preference-based NCO 中哪些监督信号更稳健、哪些校准规则更依赖分布。因此，我们将 objective discovery 视为一种 evaluated hypothesis generation：LLM 和 evolutionary operators 提出 objective-design hypotheses，semantic gate 保证这些 hypotheses 满足基本偏好语义，training-based evaluation 再决定哪些 hypotheses 真正能改善优化。

## What the Discovered Objectives Reveal

四个问题上最终存活的 objective 并不是简单地“把更优解的 log-prob 拉高、把更差解的 log-prob 压低”。更准确地说，它们都在学习一种局部比较的校准规则：policy margin 仍然是主要被优化的对象，但 objective-derived signal 只以有界、归一化、或者实例自适应的方式进入 loss。这个结果改变了我们对 preference-based NCO 的一个直觉认识：目标值差距本身并不总是适合作为线性监督强度；更稳健的做法是让它决定“比较应当多确信”，而不是直接决定“梯度应当多大”。

在 TSP 中，最终 loss 近似为

$$
\mathcal{L}_{\mathrm{TSP}}
= \frac{\sum_i w_i\left[-\log\sigma\left(\mathrm{clip}\left(\alpha s(\ell_w-\ell_l)\left(1-\beta\frac{\Delta c_i}{1+|\Delta c_i|}\right),-20,20\right)\right)\right]}{\sum_i w_i+\epsilon}.
$$

这里的关键不是使用了 cost gap，而是 cost gap 被压缩为 $\Delta c/(1+|\Delta c|)$ 后只作为 margin 的温和调制项。换言之，TSP 上有效的 objective 不是“gap 越大惩罚越大”，而是“gap 越大，policy margin 的置信校准略有改变”。这解释了为什么一些看似更直接的 cost-additive 公式会被 gate 或训练过程淘汰：它们把路径长度的尺度变化误当成了偏好强度变化。

CVRP 的最终公式更有启发性。它没有依赖原始 cost gap，而是围绕 $\Delta\mathrm{rank}$ 的均值、方差和 coefficient of variation 构造动态 bias 与 margin scale。简化来看，它学习的是

$$
x = \alpha \, s_{\mathrm{cv}}(\Delta r)(\ell_w-\ell_l)
- \beta_{\mathrm{cv}}\Delta r
+ \tanh\left(\gamma b_{\mathrm{dyn}}(\Delta r)\right),
\qquad
\mathcal{L}_{\mathrm{CVRP}}=\sum_i w_i[-\log\sigma(\mathrm{clip}(x_i,-14,14))].
$$

这说明 CVRP 的 preference signal 更依赖候选池内部的排序结构，而不是绝对 route cost。CVRP 的容量约束会让相近 cost 的候选解在结构上差异很大；因此，存活 objective 学到的是一种 pool-relative calibration：当排序分散度较高时保守缩放 margin，当排序结构稳定时才引入更强的 bias。这个发现比“CVRP 需要更大的 gap 权重”更具体，也更能解释跨问题迁移中 weighting rule 的脆弱性。

FFSP 和 JSSP 的最终 loss 则共同揭示了调度问题上的另一种规律：advantage gap 是有用的，但它更像实例级温度或归一化尺度，而不是逐 pair 的线性惩罚。FFSP 的最佳公式使用 $\exp(\mathrm{clip}(|\mathrm{mean}(A)|,0,1))-1$ 作为全局 margin 系数，并用 advantage norm 归一化 policy margin；JSSP 进一步简化为

$$
\mathcal{L}_{\mathrm{JSSP}}
= \frac{\sum_i w_i[-\log\sigma((\ell_w-\ell_l)/\max(\mathrm{mean}|\Delta A|,\epsilon))]}{\sum_i w_i+\epsilon}.
$$

这两个公式的共同点是，它们没有让每个 pair 的 advantage gap 独立地放大梯度，而是先从候选池中估计一个稳定尺度，再用这个尺度重标定所有 pairwise margins。对调度问题而言，这个设计尤其合理：makespan 或 flow-shop objective 的绝对差距受实例结构影响很大，直接使用会把“实例难度”混入“偏好置信度”。最终 objective 将两者分离开来，因此更稳定。

总体来看，discovery 的真正收获不是某一个特殊函数形式，而是一个跨问题的设计原则：有效的 preference objective 应当保持 policy-margin 主体，同时把 objective signal 约束为 bounded confidence, pool-relative scale, or instance-adaptive normalization。这个原则也解释了为什么简单的 BOPO/PO-style loss 仍然是强基线：它们的核心 pairwise margin 是对的；搜索带来的增益来自更细粒度地校准何时、以多大强度相信这些 pair。

## How Evolution Refines Objective Hypotheses

搜索轨迹显示，演化并不是随机地堆叠复杂算子，而是在逐步修正 objective hypothesis 中的三个不稳定来源。

第一步通常是建立可用的 pairwise margin 骨架。早期个体大多围绕 $-\log\sigma(\ell_w-\ell_l)$、hinge-like margin 或 rank-weighted margin 变化；这些个体能表达“winner 应被提升”的基本偏好语义，但常常缺少尺度控制。因此，它们在某些问题上能通过局部梯度检查，却在 CO-alignment gate 中暴露出 objective sensitivity 或 affine invariance 问题。

第二步是把原始 objective signal 从 additive penalty 改成 multiplicative 或 normalized calibration。TSP 的轨迹最典型：被拒绝的早期公式经常包含 $-0.1(c_l-c_w)$ 这类直接加到 logit 上的项；最终公式去掉了 additive offset，改为 bounded normalized gap 对 policy margin 的乘性调制。这一步的改进不是单纯“更平滑”，而是消除了对 cost 平移和尺度的错误敏感性，使 loss 更接近比较排序而不是绝对数值回归。

第三步是引入实例或候选池级别的统计量。CVRP 从 rank gap 发展到 rank dispersion；FFSP/JSSP 从逐 pair advantage weighting 发展到 advantage-based normalization。这个变化很关键：preference batch 中的 pairs 不是独立样本，而是同一候选池内的相对比较。演化最终学到的稳定公式都显式或隐式地尊重这一点，把监督强度放在 pool-level context 中解释。

最后，高保真评估在轨迹中起到了“反直觉过滤器”的作用。一些公式在语义上看起来更丰富，例如混入 entropy regularization、rank-probability normalizer 或多重 bias 项，但它们并没有稳定改善训练；相反，较晚存活的公式往往更克制：一个 bounded link、一个清晰的 margin、一个必要的尺度校准。这说明搜索过程的价值不只是生成新公式，也在于用训练反馈压制那些形式复杂但优化语义松散的 hypothesis。

## Why the Search Design Matters

![Gate-rejected analysis](figures/gate_rejected_analysis/fig_gate_rejected_analysis.png)

上图比较了最终接受的 best losses 与 replay 得到的 gate-rejected individuals。我们将 rejected candidates 分为 preference-direction failure、objective-insensitive failure、affine-unstable failure、runtime/numeric failure 和其他失败。四个 loss-search setting 的结果表明，被拒绝的个体不是随机的“差公式”，而是在暴露搜索设计必须过滤掉的特定失败模式。

最常见的失败并不是数值溢出，而是语义错位。在 replayed rejected set 中，preference-direction failure 的 median swap gap 为负：交换 winner 和 loser 后，loss 反而朝错误方向变化。这些公式往往仍然包含看起来合理的成分，例如 $\ell_w-\ell_l$、rank gap 或 normalized weight；但符号、bias 或聚合方式让最终 objective 奖励了错误排序。因此，单纯的语法检查或 compile-only filter 不够：许多 rejected candidates 可执行、可微分，却在梯度语义上是反的。

第二类失败是 objective insensitivity。这些个体可能有很大的正向 semantic swap gap，因此表面上确实偏好 winner；但当底层 objective perturbation 改变时，它们的 loss 几乎不变。汇总统计中，objective-insensitive group 的 median objective response 等于 0。这类公式通常被 policy log-prob、entropy-like regularizer 或饱和非线性主导。它们通过了局部偏好检查，却没有回答更深的 CO 问题：loss 是否仍然在听优化目标本身。

第三类失败是 affine instability。Affine-unstable candidates 会响应 objective difference，但也会响应无关的 objective 平移或缩放。replay 结果中，这一组具有非零 median affine drift，而 accepted best losses 的 median affine drift 为 0。公式层面的原因很直接：被拒绝的个体经常把 raw cost gap 或未归一化的 rank/cost penalty 直接加到 margin 上；被接受的个体则使用 bounded gap transform、rank dispersion 或 advantage normalization。因此，gate 并不只是拒绝“不安全代码”，而是在强制 CO preference objective 所需的不变性。

这也解释了为什么 family diversity 重要。当移除 family 机制后，后期 no-family search 并不是简单探索了“更大的空间”，而是漂移到表面新颖但语义约束更弱的变体：entropy regularizer、advantage-heavy blend、无结构的 weighted sum，以及同一类 margin-plus-extra-regularizer motif 的重复重组。在 late no-family population 中，37.5% 的公式出现额外 regularizer，而四个 accepted best formulas 均没有使用这类项。换言之，no-family search 花了更多预算让公式“看起来不同”，但不一定让 preference signal 更好地校准。Family constraints 提供的是一种保持互补假设竞争的压力：cost/rank calibration、advantage normalization、bounded link 和 aggregation choice 会继续互相竞争，而不是塌缩到一个过度变异的 lineage。

这些观察共同支持当前的两层搜索设计。Semantic gate 移除那些局部可执行但 preference-invalid 的 objectives；CO-alignment gate 移除那些忽略 objective 或过度依赖 objective 数值尺度的 objectives；high-fidelity training 再在剩余 hypotheses 中做最终选择。没有 gate，搜索会把算力浪费在训练前已经能看出失败的公式上；没有 family diversity，搜索会反复发现同一 motif 的高方差装饰版本。

## Why Pool-Aware Weighting Is Less Transferable

The second-stage search should not be interpreted as adding an ability that a free-form loss could never express. In fact, our FFSP and JSSP losses show the opposite: a discovered loss can already implement nontrivial weighting and batch-level calibration through terms such as weighted aggregation, mean advantage scale, or mean absolute advantage normalization. The real distinction is therefore not mathematical expressivity in an unconstrained programming language, but the search interface and the point in the pipeline where the rule is applied.

Loss discovery operates on the materialized pairwise table. After a preference builder creates pairs, `PrefBatch.to_pairwise_loss_batch` exposes flattened tensors such as `log_prob_w`, `log_prob_l`, `cost_gap`, `delta_rank`, `advantage_gap`, and `weight`. This interface is sufficient for learning a scalar objective kernel over already-created comparisons, and it can also learn global scalar normalizers over that table. However, it does not expose the pair coordinates themselves as first-class variables: the loss batch does not contain `b_idx`, `winner_idx`, or `loser_idx`. Thus the loss search is naturally biased toward functions of pair-local features and global reductions over the flattened pair set.

The weighting builder is exposed to a different object before flattening: the full candidate pool `feature_cache["objective"]` and `feature_cache["log_prob"]` with shape `(B,K)`, the generated pair indices `(b_idx, winner_idx, loser_idx)`, and pool-level features such as rank, instance objective dispersion, log-probability dispersion, and regret statistics. The best FFSP builder uses `instance_obj_mad[b_idx]`, `instance_log_prob_std[b_idx]`, rank span, and a tie-zone mask; the best JSSP builder uses `instance_obj_mad[b_idx]`, `instance_regret_mean[b_idx]`, and rank difference. These rules are not merely multiplying each flattened pair by a number after the fact. They define a pre-loss measure over the complete candidate-pool geometry: how much mass each pair receives depends on where the pair sits inside its source instance and how that instance's candidate pool is distributed.

This reframing also explains the transfer behavior without contradicting the FFSP loss. The loss stage can discover robust comparison kernels and table-level calibrations, including batch-level statistics. The weighting stage searches a pool-aware data valuation rule tied to the source distribution's rank gaps, objective spread, regret scale, and policy margin spread. Such rules can improve an in-distribution setting because they exploit the geometry of the candidate pools produced by that policy and problem size. But the same geometry is less stable across problem families or scales: a rank span, normalized gap, or regret scale that identifies informative pairs on FFSP-100 may correspond to a different training signal after transfer. In this sense, second-stage weighting is better described as pool-aware pair-measure search, not as a universally more expressive loss search.

The checkpoint-rollout view makes this failure mode more concrete. For each trained checkpoint, we sort generated winner-loser trajectory pairs by their actual $\Delta$cost, defined as $c_{\mathrm{loser}}-c_{\mathrm{winner}}$, and plot the weight assigned by the discovered builder. This exposes weighting as a scale-calibrated pair measure rather than an invariant semantic kernel. In CVRP, moving from CVRP100 to CVRP50 shifts the rollout $\Delta$cost distribution downward, so a raw-gap weighting rule leaves more pairs near the lower-clamp region. In TSP, the rule quickly saturates at the upper clamp, making medium and large $\Delta$cost pairs almost indistinguishable. FFSP shows an even sharper distinction from a simple gap-strength story: because the rule combines tie-zone, normalized margin, and inverse rank-span terms, larger $\Delta$cost pairs can receive lower weight. JSSP expands the raw $\Delta$cost range from 10x10 to 15x15 and pushes more pairs toward the upper clamp. Thus the transfer weakness is not a failure of pairwise preference semantics; it is a failure mode of the learned sampling measure. Weighting can improve the source task by allocating mass to the useful regions of the source checkpoint's pair distribution, but that allocation rule need not remain stable when the generated trajectory distribution changes.

The contrast between CVRP and FFSP is clearest on a second cost-free pair coordinate: normalized policy margin, $|\ell_w-\ell_l|/\sigma_\ell$, where $\sigma_\ell$ is computed within the same checkpoint-generated candidate pool. This axis is available to both tasks and does not depend on the unit of the objective. CVRP weighting is almost flat along this axis because its builder is essentially a cost-gap rule; FFSP weighting has a strong negative slope because its builder contains an inverse normalized-margin factor. Therefore the difference is not merely that the two tasks have different cost units. The searched weighting rules encode different sampling measures over the same all-pairs topology: CVRP mainly allocates mass by objective separation, while FFSP allocates mass to low-margin, locally ambiguous comparisons.

For FFSP, however, the final-weight curve alone understates the mechanism because the rule is heavily clipped. The pre-final score divides a margin-based term by rank span and then applies an upper clamp. On checkpoint rollouts, roughly half of the FFSP pairs exceed the upper clamp, and this upper-clipped mass is concentrated on low-margin pairs. Consequently, FFSP does not show a large source-to-target shift in the marginal final-weight histogram: much of the relevant variation has already been collapsed to the same maximum weight. This gives a different kind of transfer risk from CVRP. CVRP exposes raw-gap scale drift, whereas FFSP exposes saturation of a low-margin pair selector.

To expose the FFSP scale effect, it is more informative to look at the raw policy margin $|\ell_w-\ell_l|$ before instance-level normalization. In our checkpoint replay, the median raw margin drops from about 14.6 on FFSP100 to about 7.6 on FFSP50, while the 90th percentile drops from about 36.5 to about 18.7. Thus FFSP50 is not distinguished by a different final-weight histogram; it is distinguished by a compressed policy-margin signal that is then normalized and clipped by the weighting rule.
