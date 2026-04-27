### Stage 1: Loss Search

#### Problem formulation

在第一阶段中，我们搜索一个可执行的候选损失程序 $$f_\phi$$。给定当前策略参数 $$\theta_0$$、固定参考 builder $$g_{\mathrm{ref}}$$ 以及有限训练预算 $$T_{\mathrm{patch}}$$，候选损失 $$f_\phi$$ 通过短程训练得到更新后的模型参数

$$
\theta'(\phi)=\mathrm{Train}\big(\theta_0; f_\phi, g_{\mathrm{ref}}, T_{\mathrm{patch}}\big).
$$

随后，我们在验证集上评估该模型的表现，并将其作为候选损失的适应度：

$$
\mathcal{F}(\phi)=\mathrm{ValMetric}\big(\theta'(\phi)\big).
$$

由于当前实验的目标是最小化验证代价，因此搜索目标为

$$
\phi^\star=\arg\min_\phi \mathcal{F}(\phi).
$$

这一表述对应于一种外层 objective search、内层短程训练的双层优化视角。与直接比较候选 loss 的解析形式不同，我们真正关心的是某个损失是否能够在有限训练预算内有效驱动策略学习。因此，第一阶段搜索的不是静态 loss 数值，而是 loss 所诱导出的训练动力学。

在当前方法中，第一阶段采用 `search_mode=loss_only`。这意味着 builder 侧被冻结，系统只搜索损失函数，而 pair construction 固定为参考 builder。与此同时，我们采用 `llm_init_only=true`，因此初始 loss population 完全由 LLM 生成，而不是由手工基线或预置模板混合初始化。

#### Search space of candidate losses

为了让自动发现过程不被人工模板限制，我们将候选losses定义为一个程序。具体地，搜索器输出的每一个候选都需要实现一个 `forward(batch)` 风格的标量函数，其输入是一组由 preference batch 提供的张量特征。

这些输入特征主要包括三类：

1. 目标值相关信号，用于显式反映 winner 与 loser 之间的质量差异，例如 cost、cost gap 或其归一化形式。
2. 概率相关信号，用于描述当前策略对 winner 与 loser 的偏好程度，例如 `log_prob_w`、`log_prob_l` 及其差值。
3. 可选辅助观测量，用于扩展搜索空间，例如 advantage、rank gap、归一化统计量或其他稳定化量。

为了保证候选程序可编译、可控且适合 GPU 上的批量计算，我们显式约束了可用算子集合。允许使用的算子覆盖单调变换、数值稳定变换、张量聚合、基本代数运算以及若干归一化/裁剪操作。这样做的目的是在保持表达力的同时，确保候选目标始终处于可验证、可执行的程序空间中。

#### COP-aware structural and semantic gating

如果直接允许 LLM 在上述程序空间中自由生成 loss，大量候选都会是无效的：有些会把 winner 与 loser 的语义写反，有些在少量样本上看似可运行，但一旦进入反向传播就会出现梯度方向错误或数值爆炸。因此，在真正进入训练前，我们引入了分层门控机制。

第一类是结构与运行时安全门控。系统会检查候选能否成功编译、能否完成前向与反向传播、是否存在非有限 loss、缺失梯度或其他数值异常。在当前配置中，还额外加入了 sandbox gate，对候选程序进行隔离执行；当 sandbox 判定失败时，该候选会被阻止进入后续高保真训练。

第二类是偏好语义与联合梯度门控。这里关注是“该 loss 是否真的表达了正确的偏好优化语义”。核心检查包括：

2. 交换一致性：交换winner与loser后，loss的行为应与偏好方向同步翻转，而不能保持原状或给出错误奖励。
3. 局部梯度方向：winner的log-prob 应倾向于被提升，loser的log-prob应倾向于被压低；若这一点在局部梯度上就不成立，则该目标不具备稳定的优化语义。

第三类是CO-alignment门控。系统会进一步检查候选loss是否对组合优化目标保持合理的几何结构，例如是否对纯粹的平移/缩放过度敏感、是否在目标 gap 增大时仍能维持一致的比较驱动力等。这些检查共同构成面向COP的objective prior，使搜索真正聚焦于“如何更好地利用objective signal与policy signal”。

在候选未通过上述门控时，系统不会立刻丢弃，而是会调用 repair 流程进行自动修复。对于当前配置，这一 repair 机制是方法的一部分：系统会针对 forward error、joint preference violation、数值异常以及 sandbox 失败等问题进行多轮修复，只有在修复仍然失败时才终止该候选。

#### High-fidelity fitness estimation

通过门控的候选将进入高保真短程训练评估。为了加快评估个体的效率，高保真评估是一个带promotion的分轮筛选过程。当前配置包含两个高保真round：

1. 第一个round使用较低预算对全部通过门控的候选进行短程训练与验证。
2. 第二个round仅对第一轮中表现更优的候选继续训练，并使用更高预算进行更严格的比较。

#### Search algorithm: a COP-oriented adaptation of EoH

为了在上述受约束程序空间中高效搜索，我们对演化式搜索框架做了面向COP objective discovery的改造。其核心思想是将LLM嵌入一个带有family diversity与 automatic repair的进化循环中。

具体而言，候选并不是通过单一prompt生成，而是通过一组语义角色不同的操作符产生。针对loss search，我们使用的主要操作符包括：

- `GEN`
- `PARADIGM_SHIFT`
- `STRUCTURE_SHIFT`
- `CONSTRAINT_INJECT`
- `XOVER`
- `TUNE`

其中，`GEN` 负责从头提出新目标；`PARADIGM_SHIFT` 尝试跨越现有范式重写 loss；`STRUCTURE_SHIFT` 调整已有目标的表达结构；`CONSTRAINT_INJECT` 注入归一化、温度、边界或稳定性约束；`XOVER` 在两个候选之间执行结构性交叉；`TUNE` 对现有表达式的局部敏感组件进行微调。当前配置对这些操作符的预算并不平均，我们显著偏向于结构变化与约束注入，从而优先探索“目标形式的变化”，并少量进行局部参数扰动。

此外，我们启用了family diversity机制，在elite集合与parent pool中保留不同family的候选，同时限制单一family的占比，以避免搜索过早坍缩到某一种loss family。配合repair机制，第一阶段形成了一个面向组合优化目标发现的、以高保真训练效果为最终导向的演化搜索过程。

### Stage 2: Pair Weighting

#### Problem formulation

在第一阶段得到固定损失函数 $$f^\star$$ 之后，第二阶段不再修改比较目标本身，而是进一步搜索pair weighting规则 $$g_\psi$$。设对于实例 $$x$$，当前策略生成的候选解集合为

$$
\mathcal{Y}(x)=\{y^{(1)},\dots,y^{(N)}\}.
$$

在本阶段中，我们固定pair construction为由解质量排序诱导得到的full pair集合，因此winner-loser的配对关系固定，搜索器唯一需要决定的是，每一个有效偏好对在训练中应当被赋予多大的权重。

具体地，builder 输出

$$
\mathcal{P}_\psi(x)=\{(y_w^{(i)},y_l^{(i)},w_\psi^{(i)})\}_{i=1}^{M(x)},
$$

其中 $$(y_w^{(i)},y_l^{(i)})$$ 表示固定 full pair construction 下的第 $$i$$ 个有序偏好对，$$M(x)$$ 为实例 $$x$$ 上的有效偏好对数目，$$w_\psi^{(i)}\ge 0$$ 为weighting rule $$g_\psi$$ 为该 pair 分配的权重。由于pair geometry已经固定，第二阶段只改变这些比较在损失聚合中的相对贡献。

给定固定损失 $$f^\star$$，第二阶段的优化目标为

$$
\psi^\star=\arg\min_\psi \mathcal{F}_{\mathrm{weight}}(\psi;f^\star),
$$

其中

$$
\mathcal{F}_{\mathrm{weight}}(\psi;f^\star)=\mathrm{ValMetric}\!\left(\mathrm{Train}(\theta_0; f^\star, g_\psi, T_{\mathrm{patch}})\right).
$$

#### Transfer from Stage 1 to Stage 2

两阶段并不是概念上松散串联，而是通过显式工件完成衔接。第二阶段启动时，会继承第一阶段得到的最优loss，并以此初始化loss population。在第二阶段loss侧在整个搜索过程中保持冻结，系统只在builder侧进行演化。因此，Stage 2 可以被理解为“在固定最佳比较规则后的 importance allocation search”。

#### Reweight-only search space

相比于第一阶段的 loss search，第二阶段的搜索空间更受约束。

这一约束在实现中是硬性的。系统会用固定模板 builder 生成参考 pair index，并检查候选 builder 生成的 pair index 是否与参考模板完全一致；如果候选改变了 pair 的结构、数量或顺序，则直接判为非法。换言之，第二阶段搜索的不是一般意义上的 builder，而是一个在固定 full-pair 结构上的 weighting program。

#### Candidate weighting families

在当前配置中，第二阶段的候选 weighting family 包括：

1. `uniform_none`：不引入显式权重，即所有 pair 等权。
2. `gap_linear`：权重与 winner-loser 的 objective gap 线性相关。
3. `gap_softmax`：对 objective gap 进行温度控制后的指数型加权。
4. `gap_sigmoid`：使用 sigmoid 将 objective gap 映射到平滑权重。
5. `gap_square`：使用平方 gap 强调大差距 pair。

设某个 pair 的目标差距为

$$
\Delta_i = c_l^{(i)} - c_w^{(i)},
$$

则第二阶段首先根据 weighting family 计算原始权重 $\tilde w_i$，再在每个实例内部做归一化，得到最终用于训练的 $$w_\psi^{(i)}$$。这一设计避免了 weighting 规则仅通过放大或缩小整体 loss scale 来影响优化过程，使其作用集中体现在“如何重分配同一实例内不同 pair 的相对重要性”上。

#### Why fixing full pair and searching only weighting

一个自然的替代方案，是在第二阶段同时搜索 pair construction 与 pair weighting。然而，在当前训练 pipeline 中，计算资源的主要开销集中在 rollout 阶段，即生成候选解集合 $$\mathcal{Y}(x)$$ 的过程，而不是 rollout 之后的 pair 构造与聚合。相较之下，减少 pair 数量所带来的额外计算节省有限，却会直接损失可用于监督的相对排序信息。

因此，我们在第二阶段固定采用信息最充分的 full pair construction，并仅对 weighting rule 进行搜索。这样的设计有两点直接好处：

1. 它完整保留了同一实例内候选解之间的相对优劣关系，使训练能够利用最丰富的偏好监督信号。
2. 它将搜索自由度集中在 pair importance 上，使算法能够在不牺牲监督覆盖度的前提下，自适应地强调更有训练价值的比较关系。

因此，第二阶段可以被理解为一个轻量但有效的数据估值过程：它不是通过删减 supervision 来降低复杂度，而是在保留 full pair 信息的基础上，进一步学习哪些偏好对更值得被重点利用。

#### Evaluation in Stage 2

与第一阶段不同，第二阶段恢复了多保真评价流程。由于当前配置未关闭 proxy，因此候选 weighting builder 会先经过 cheap gate 与 proxy evaluation，再由 proxy 分数筛选出更有前景的候选进入高保真短程训练。最终，系统以高保真评估结果更新 builder population，并输出与固定损失 $$f^\star$$ 对应的最优 weighting rule。

综上，整个方法形成了一个清晰的两阶段分解：第一阶段先搜索“如何比较”候选解，即发现更优的 preference loss；第二阶段再在固定最优 loss 的前提下搜索“应优先强调哪些比较”，即发现更优的 pair weighting 规则。前者决定 comparison rule，后者决定 importance allocation rule，两者共同构成最终的 preference-based training objective。
