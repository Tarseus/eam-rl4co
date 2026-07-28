# RFPS 自适应几何步长：结果与方法结论

## 1. Čencov 唯一性给出了什么

在正概率单纯形

`Delta_M^o = {q_m > 0, sum_m q_m = 1}`

上，切空间为 `T_q Delta = {u: 1^T u = 0}`。要求度量在充分统计或相容
Markov 映射下保持不变时，Čencov 定理把度量唯一确定为 Fisher--Rao 度量的
正常数倍：

`g_q(u,v) = lambda * sum_m u_m v_m / q_m`。

这足以唯一确定 RFPS 应使用的方向概念，却不能唯一确定绝对步长，因为常数
`lambda` 与 trust radius 可以互相吸收。

## 2. 当前投影向量其实已经是 Fisher 自然梯度

令 `c_m = -partial L / partial p_m`，并有
`p_m(q)=p_m^0+log(q_m/q_m^0)`。固定当前 q 已确定的配对和统计量时，

`partial L / partial q_m = (partial L / partial p_m) / q_m`。

把该余切向量用 Fisher 逆度量升为切向量，得到负黎曼梯度

`u_m = c_m - q_m * sum_j c_j`，

恰好就是 RFPS v4 已使用的中心化向量。需要修正的不是 `u`，而是其单位化、
第二点构造和跨点比较方式。

## 3. 不变的 d0、q1 和 d1

定义 Fisher 范数

`||u||_q = sqrt(sum_m u_m^2/q_m)`，

并使用平方根嵌入 `h=sqrt(q)`。则单位球面切向量为

`d(q) = u / (sqrt(q) * ||u||_q)`。

第一方向是 `d0=d(q0)`。若选 Fisher--Rao 弧长 `ell`，一次指数映射为

`h1 = cos(ell/2) h0 + sin(ell/2) d0`，

`q1 = h1^2`。

在 q1 重算完整程序对损失和自然梯度，得到 `d1_raw=d(q1)`。由于 `d0` 和
`d1_raw` 属于不同切空间，应沿 h0--h1 的球面测地线把 d1 平行移动回 h0：

`d1 = PT_{h1->h0}(d1_raw)`。

最终每个 probe 的描述符仍只有 `[d0 || d1]/sqrt(2)`。这不是恢复旧版三点
曲率系统；它只有一次指数映射、一次闭式平行移动和两次损失求导。

## 4. 为什么度量本身不能自动产生候选级步长

当前 q0 均匀，且 log-weight 方向满足 `1^T d=0, ||d||_2=1`。对
`q_eta=softmax(log q0+eta*d)`，

`KL(q0||q_eta) = log[(1/M) sum_m exp(eta d_m)]
                 = eta^2/(2M) + O(eta^3)`。

所以固定 KL/Fisher 半径给出

`eta approximately sqrt(2M delta_KL)`，

在二阶上与候选无关。M=100、eta=0.03 对应
`delta_KL approximately 4.5e-6`。反向 KL、熵下降和 ESS 在该起点也具有相同
二阶退化。

## 5. 实验结果

固定 KL、反向 KL 和 ESS 的一维精确求根都返回 0.03000，步长 CV 为
0--0.002，所有指标与固定步长相同。候选级自适应结果如下：

| 规则 | 步长 CV | 结果 | 决定 |
|---|---:|---|---|
| 固定 forward KL | 0--0.000 | 与 eta=0.03 相同 | 可作几何重参数化 |
| 固定 reverse KL / entropy | 0--0.001 | 无增益 | 删除 |
| 固定 ESS | 0.001--0.002 | 无增益 | 删除 |
| max log-ratio | 0.143--0.159，median 约 0.05 | rho、NN、误跳过均无稳定增益 | 只可作数值上限 |
| turning radius | 0.380--0.569 | rho 无增益，个别 warm NN 恶化且多一次求导 | 删除 |

真正的 Fisher 两点方向对在相同 0.03 弧长下相对欧氏方向对有一致的小幅 rho
增益：matched scratch 为 .808/.816 对 .806/.810，matched warm 为
.856/.863 对 .853/.855，external 为 .714/.716 对 .712/.715；误跳过率不变。
因此 Fisher 的合理作用是给 d0、d1 和跨点比较一个不变定义，而不是制造
候选级曲率步长。

## 6. 推荐的自动标定

采用 `lambda=1` 的 Fisher 度量约定，但不手填 ell。对参考 checkpoint 的真实
优化器做一次 stateless 更新，在相同固定轨迹上重放动作，得到

`Delta p_m^train = log pi_{theta+}(tau_m) - log pi_theta(tau_m)`。

令

`q_train = softmax(log q0 + center(Delta p^train))`，

并测量真实一步在当前经验流形上的 Fisher--Rao 距离

`ell_train = 2 arccos(sum_m sqrt(q0_m q_train_m))`。

在若干参考 batch 和标准/当前精英程序对上取稳健中位数 `ell_star`。RFPS 对
所有候选使用相同 `ell_star`。这样做具有三点好处：

1. 步长单位来自实际训练算法、学习率、Adam 状态和网络敏感性；
2. 候选描述符仍对损失的纯常数缩放保持稳健；
3. 在线候选仍只需要两次表达式求导，不增加网络反向。

如果希望每个候选拥有不同训练步长，就必须为每个候选计算 stateless 网络
更新或模型 Jacobian 的 pullback tangent。此时更合理的是直接使用该网络可达
切向量，而不只是给当前 d0 乘一个候选标量；这会成为更昂贵的新描述符，需要
单独评估成本收益。

## 7. 最终建议

把 RFPS v4 的欧氏方向对改为轻量的 Fisher-invariant two-point response：

- d0、d1 使用 Fisher 单位切向量；
- q1 使用一次 Fisher 指数映射；
- d1 闭式平行移动回 q0；
- 弧长 ell 由一次真实优化器更新离线标定；
- 不恢复多步积分、三点曲率、ESS 通道或 candidate-wise turning radius。

这保留了几何的理论必要性，同时仍是一个单链、两次求导的方法。
