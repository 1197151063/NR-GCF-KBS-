# 未解决问题与下一步决策

## 最高优先级：outputs_v1.8 clean safety 与 momentum 排序消融

只新增：

- clean + no filtering；
- clean + early fused filtering；
- 20% noise + 同一 momentum-calibrated budget、`structure_weight=1.0` 排序。

第三组需与 outputs_v1.7 的 20% noise、`structure_weight=0.95`、
`original_always` 比较。若 0.95 稳定更好，才能把 5% momentum 写成推荐排序贡献；
若两者相同或 1.0 更好，则 momentum 暂时只保留为删除预算校准信号。

## 已完成：outputs_v1.7 early timing pilot

只验证 20% uniform degree-preserving replacement noise、seed 2026：

- `original_always` + no filtering；
- `original_always` + adaptive epoch 2–4 filtering；
- `reliability_weighted_always` + 同一 early adaptive filtering。

审阅顺序：

1. commit、noise ratio、配置是否正确；
2. 三组过滤前轨迹是否完全相同；
3. adaptive 是否在 epoch 3 左右触发，以及两个 filtering arm 的 removed edges 是否相同；
4. filtering 相对 no-filter 是提升还是伤害；
5. weighted RMS 是否在过滤后实际不同，能否减小触发当轮冲击；
6. 修正后的 best post-filter Recall@20 是否提高；
7. overall best 是否从过滤前移到过滤后，final epoch 是否稳定而非偶然峰值。

本轮不改变删除预算。只有确认 early filtering 的方向后，才决定是否降低预算。

## 若 reliability-weighted always 有效

下一步不要继续调很多超参数，优先：

1. 固定 adaptive timing 规则，不再按测试指标挑选 T；
2. 补一个 clean/noisy 的同配置 baseline；
3. noise ratio 扩到 0.1、0.2，暂时仍只用 seed 2026；
4. 在第二数据集复现趋势；
5. 最后才补 2–3 seeds；
6. 记录运行时间和显存。

## 若 weighted 只改善 post-filter、但 overall best 仍在 filtering 前

可能原因：

- filtering time 太晚，相对于当前更快收敛的 `init_weight=0.01/lr=5e-4` 配置；
- T=20 是在旧配置下选择的，不能无条件迁移；
- stage-two 训练时间不足或图切换产生适配问题。

此时只需要重新比较 T=10/15/20 中极少数点，但必须基于 always-on norm 和新配置，不能复用旧时间结论。

## 若 weighted 与 original 完全相同

先检查 trace：

- node confidence 是否几乎常数；
- weighted 与 unweighted RMS 是否数值相同；
- policy 是否成功传入 model buffers；
- filtering 后 mode 是否仍为 always active。

如果 node confidence 被高 degree 邻域平均得过于平滑，可考虑但尚未实现：

- node confidence 使用低分位数而非均值；
- 使用 reliability power/temperature 增强差异；
- 分别使用 user-side 和 item-side confidence；
- 仅用高风险 retained tail 调整尺度。

不要未经诊断直接增加复杂网络或新 loss。

## 若 weighted 性能下降

优先结论：global scale calibration 不适合直接由当前 edge reliability 加权。

可保留：

- hard structure-momentum filtering；
- original always-on cross norm。

不要为了统一叙事强行保留 weighted modulation。论文可以把两个部分描述为共享可靠性动机，但必须由实验证据决定是否耦合。

## 仍需补的关键消融

- 同删除预算的 pure structure vs 95/5 fused risk。
- original always + no filtering。
- original always + hard structure-only。
- original always + hard structure-momentum。
- original always vs reliability-weighted always。
- overall best vs best post-filter。

## 原实现歧义

必须最终决定论文/代码采用哪种 cross norm：

- 用户提供的 direct `x=cross_norm(x)`；当前 KBS 代码采用此版本。
- 原 Git commit 的 `lambda*x_c+(1-lambda)*x`；原日志 config 中 lambda=0.6。

若需要精确复现实验日志，必须使用对应 commit 验证实际 forward，不能仅根据打印 config 推断。
