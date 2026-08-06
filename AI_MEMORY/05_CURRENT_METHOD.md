# 当前方法设计

## 暂定名称

可暂称：Structure-Dominant Edge Reliability with Reliability-Calibrated CrossNorm。

不要现在锁定论文最终名称和最终融合公式。

## 第一部分：Edge reliability

对每条训练边 `e=(u,i)`：

### Stable training dynamics

维护 detached instance BPR loss EMA：

`m_e(t) = 0.9*m_e(t-1) + 0.1*l_e(t)`。

第一次出现时直接令 `m_e=l_e`。

### Bilateral structure

- `s_user(e)`：item `i` 与 `N(u)\{i}` 中其他 item 的二跳一致性。
- `s_item(e)`：user `u` 与 `N(i)\{u}` 中其他 user 的二跳一致性。
- `s(e)`：可用两侧的均值。

### Fused risk

对训练边做 percentile rank：

`risk(e) = 0.95*(1-rank(s(e))) + 0.05*rank(m_e)`。

结构信号占主导，momentum 只用于修正排序。

### Adaptive budget

`B = count(rank(m_e)>=0.8 AND rank(s(e))<=0.2)`。

删除 fused risk 最高的 B 条边。不使用 degree protection。

### Adaptive filtering time

CrossNorm 从 epoch 1 始终开启，filtering 时点由 edge reliability readiness 决定，而不是由 norm 的开关决定：

- 最早检查 epoch 5；
- 已观察 edge 比例至少 0.99；
- 相邻 preview removed sets 的 Jaccard 至少 0.90；
- 连续两次满足稳定条件后触发，理论最早 epoch 7；
- 若未稳定，epoch 10 强制触发；
- preview 只读取训练 edge loss/structure，不读取 Recall、validation/test 或 synthetic label；
- 未观察 edge 的 momentum rank 为中性 0.5，且不进入 high-momentum 删除预算。

二跳结构特征只计算一次并在 readiness checks 之间缓存。每轮 preview 不修改传播图、sampler、optimizer、参数或随机数状态。

`outputs_v1.6` 中该规则在 epoch 7 触发，但 noisy overall best 已在 epoch 4。
因此 `outputs_v1.7` 暂时测试更早的 min=2、max=4、stable checks=1。这个窗口
只是一轮 timing pilot，不是当前已验证的最终方法配置；删除预算保持不变。

### Hard graph update

删除后同时改变：

- LightGCN propagation graph；
- BPR positive sampler；
- normalized adjacency。

evaluation mask 保持过滤前完整 observed train edges。

## 第二部分：Always-on original cross norm

按用户明确指定的实现，每层传播后直接执行：

```text
x = propagate(A, x)
x = cross_norm(x)
```

无权 cross norm：

`r_U = sqrt(eps + mean_u ||x_u||^2)`

`r_I = sqrt(eps + mean_i ||x_i||^2)`

`x_u <- x_u / r_I`

`x_i <- x_i / r_U`

它从 epoch 1 开启，作为 backbone 的 layer-scale calibration。

当前 direct 模式不使用 lambda 混合。

## 第三部分：Reliability-Weighted Cross Modulation

在 filtering point 冻结 edge reliability `c_e=1-risk(e)`。

只在 retained graph 上聚合节点置信度：

`q_u = mean_{e incident to u, retained} c_e`

`q_i = mean_{e incident to i, retained} c_e`

缺失 edge reliability 视为中性 1；无 retained incident edge 的节点权重为 0。

过滤后改用：

`r_U_rel = sqrt(eps + sum_u q_u||x_u||^2 / sum_u q_u)`

`r_I_rel = sqrt(eps + sum_i q_i||x_i||^2 / sum_i q_i)`

随后仍然直接交叉缩放：

`x_u <- x_u / r_I_rel`

`x_i <- x_i / r_U_rel`

## 为什么这个衔接自然

- filtering 处理最不可靠边的离散拓扑影响；
- retained edge 中仍有未被删除的残余噪声；
- reliability-weighted RMS 让高置信节点更多决定全局尺度；
- cross norm 的 operator 从 epoch 1 到最后始终存在，不发生突然开关；
- filtering 后只更新 scale estimator，形成连续的可靠性利用链；
- 不增加新的 loss、可学习参数或 label leakage。

## 模式定义

- `original_always`：从 epoch 1 到结束使用无权 direct cross norm。
- `reliability_weighted_always`：过滤前与 original 完全相同，过滤后切换为 reliability-weighted RMS。
- `original_stage_two`：过滤前无 norm、过滤后 direct norm，仅作为失败/消融模式。
- `reliability_weighted_stage_two`：过滤前无 norm、过滤后 weighted norm，仅作消融。
- `none`：完全无 norm。
- `paper_stage_two`：旧命令兼容 alias，不应作为当前方法名称。

## 复杂度

- edge-to-node confidence：filtering 时一次 `O(E)` 时间、`O(U+I)` 空间。
- 每层 weighted RMS：`O((U+I)d)`，与原 cross norm 同阶。
- 不增加 dense adjacency 或 node-node matrix。
- 不增加随机采样。

## 当前尚不能声称的内容

- 不能声称 reliability-weighted modulation 已经提高最终推荐性能；`outputs_v1.6` 只显示它缓和 filtering 触发当轮的性能冲击，随后 original 很快追平。
- 不能声称两个信号存在特定 Spearman 负相关，除非实际计算。
- 不能声称论文 Eq.10 正确；实际采用的是用户指定/代码有效的 uncapped RMS。
- 不能声称第二阶段贡献 overall best，必须看 best post-filter。
