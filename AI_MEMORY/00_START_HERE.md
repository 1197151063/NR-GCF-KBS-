# NR-GCF KBS 项目记忆入口

- 最后更新：2026-08-05
- 模型代码基线：`b30a463`
- 工作仓库：`https://github.com/1197151063/NR-GCF-KBS-.git`
- 主要远程目录：`/root/cyj/NR-GCF-KBS-/code`
- 远程输出根目录：`/root/autodl-tmp/outputs`

## 新 session 的推荐阅读顺序

1. 本文件：了解当前结论和不能再走的弯路。
2. [`01_RESEARCH_SCOPE.md`](01_RESEARCH_SCOPE.md)：研究目标、禁止事项和评价原则。
3. [`05_CURRENT_METHOD.md`](05_CURRENT_METHOD.md)：当前正在验证的方法及数学定义。
4. [`04_EXPERIMENT_HISTORY.md`](04_EXPERIMENT_HISTORY.md)：所有重要实验结果和已失败方案。
5. [`07_OPEN_QUESTIONS.md`](07_OPEN_QUESTIONS.md)：下一步应做什么。
6. 需要改代码时再读 [`02_CODEBASE_AND_TRUE_BEHAVIOR.md`](02_CODEBASE_AND_TRUE_BEHAVIOR.md) 和 [`03_DIAGNOSTICS_AND_NOISE.md`](03_DIAGNOSTICS_AND_NOISE.md)。
7. 需要在服务器运行时读 [`06_REMOTE_RUNBOOK.md`](06_REMOTE_RUNBOOK.md)。
8. 新建 session 时可直接复制 [`08_NEW_SESSION_PROMPT.md`](08_NEW_SESSION_PROMPT.md) 中的提示词。

## 当前最重要的状态

- 已验证：Yelp2018、20% uniform degree-preserving replacement noise 下，结构一致性是强噪声信号。
- 已验证：稳定 per-edge EMA BPR loss 能补充结构信号；T=20 比 T=10/15 更好。
- 当前 hard filtering 风险：

  `0.95 * low_structure_rank + 0.05 * high_momentum_rank`。

- 当前删除预算：高 momentum 20% 与低 structure 20% 的交集数量；按 fused risk 从高到低删除同样数量。
- 不使用 degree/connectivity protection。这是项目明确选择，不要擅自加回。
- 原论文所谓 representation modulation 不应被机械地改成“前20轮完全关闭、过滤后突然打开”。实验表明 cross norm 更像 backbone 稳定器。
- 当前推荐衔接：cross norm 从 epoch 1 始终启用；过滤后仅把无权 RMS 统计切换为 reliability-weighted RMS。
- 当前代码主对照：`original_always` 与 `reliability_weighted_always`。
- 当前代码严格采用用户指定的 direct cross norm；`lambda_` 在这些 NRGCF 模式中不参与 forward。
- 下一轮实验尚未完成：`outputs_v1.5`，noise ratio `0` 和 `0.2`，seed `2026`。

## 必须避免的误读

- 原始日志的 `Recall@20=0.0679` 使用 `init_weight=0.01`、`lr=5e-4`，不能与默认 `init_weight=1`、`lr=1e-3` 的实验直接比较。
- 该原始日志最佳 epoch 是 19；如果只报告 overall best，可能完全没有使用 filtering 后的方法。
- 原始仓库的 filtering 只修改局部 `train_edge_index`，没有可靠地重建 `model.edge_index` 和训练 sampler。因此原始高分不是 edge denoising 有效性的证据。
- 曾经照搬论文的 `min(mean_norm_squared, 1)`，导致 divisor 几乎恒为 1，三个 modulation 模式完全相同。该实验无效，不得引用为 modulation 无效的证据。
- `origin/main` 是原始 NRGCF 仓库；`kbs/main` 才是本项目开发分支。服务器应从 NR-GCF-KBS- 仓库拉取。

## 当前代码版本的验收标准

- 三个传播层都要在 summary trace 中出现实际 RMS divisor。
- `original_always` 从 epoch 1 起 `effective_strength=1`。
- `reliability_weighted_always` 从 epoch 1 起也为 1；过滤前 RMS 与 original 相同，过滤后因 node confidence 加权而出现差异。
- 两个模式在过滤前必须得到相同 momentum、structure、删除预算和 removed edge IDs。
- 报告必须同时比较 overall best、best post-filter 和 final epoch，而不是只看 overall best。
