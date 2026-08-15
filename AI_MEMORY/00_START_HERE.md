# NR-GCF KBS 项目记忆入口

- 最后更新：2026-08-14
- 模型代码基线：以当前 `main` HEAD 为准；AI_MEMORY 初始快照建立于 `b30a463` 之后。
- 工作仓库：`https://github.com/1197151063/NR-GCF-KBS-.git`
- 主要远程目录：`/root/cyj/NR-GCF-KBS-/code`
- 远程输出根目录：`/root/autodl-tmp/outputs`

## 新 session 的推荐阅读顺序

1. 本文件：了解当前结论和不能再走的弯路。
2. [`09_EXPERIMENT_LEDGER.md`](09_EXPERIMENT_LEDGER.md)：当前最高优先级的实验账本、参数与有效性边界。
3. [`01_RESEARCH_SCOPE.md`](01_RESEARCH_SCOPE.md)：研究目标、禁止事项和评价原则。
4. [`05_CURRENT_METHOD.md`](05_CURRENT_METHOD.md)：当前正在验证的方法及数学定义。
5. [`04_EXPERIMENT_HISTORY.md`](04_EXPERIMENT_HISTORY.md)：所有重要实验结果和已失败方案。
6. [`07_OPEN_QUESTIONS.md`](07_OPEN_QUESTIONS.md)：下一步应做什么。
7. 需要改代码时再读 [`02_CODEBASE_AND_TRUE_BEHAVIOR.md`](02_CODEBASE_AND_TRUE_BEHAVIOR.md) 和 [`03_DIAGNOSTICS_AND_NOISE.md`](03_DIAGNOSTICS_AND_NOISE.md)。
8. 需要在服务器运行时读 [`06_REMOTE_RUNBOOK.md`](06_REMOTE_RUNBOOK.md)。
9. 新建 session 时可直接复制 [`08_NEW_SESSION_PROMPT.md`](08_NEW_SESSION_PROMPT.md) 中的提示词。

## 2026-08-14 当前快照

- 现有 clean/noisy、Filter、Norm、objective 和超参数证据已经冻结到 [`../experiment_archive/2026-08-14/README.md`](../experiment_archive/2026-08-14/README.md)。
- 当前启动的是 BPR 主 noise curve：LightGCN 对比完整 Filter+Norm，四个数据集、六个 noise ratio，共 48 runs；四臂组成消融随后补充。
- 当前 profile 位于 [`../configs/full_bpr_edge_filter_norm.json`](../configs/full_bpr_edge_filter_norm.json)，运行器为 [`../code/run_full_edge_filter_norm_bpr.sh`](../code/run_full_edge_filter_norm_bpr.sh)。
- Yelp/Amazon 使用 `mu=0.4, w_s=0.6`；LastFM/ML-1M 使用 `mu=0.2, w_s=0.95`，删边 cap 分别为 4%/0.5%。
- SSM 已有完整 Filter+Norm 代码接口与 48-case 运行器，但实验结果尚未返回；AU 仍只是无过滤 CrossNorm compatibility。
- 下方“当前最重要的状态”保留 2026-08-06 的历史演化上下文；若与 `09_EXPERIMENT_LEDGER.md` 冲突，以后者为准。

## 2026-08-06 历史快照（仅用于追溯）

- 已验证：Yelp2018、20% uniform degree-preserving replacement noise 下，结构一致性是强噪声信号。
- 已验证：稳定 per-edge EMA BPR loss 能补充结构信号；旧配置下 T=20 比 T=10/15 更好，但该结论不直接适用于 epoch 1 开启 CrossNorm 的快速收敛配置。
- 当前 hard filtering 风险：

  `0.95 * low_structure_rank + 0.05 * high_momentum_rank`。

- 当前删除预算：高 momentum 20% 与低 structure 20% 的交集数量；按 fused risk 从高到低删除同样数量。
- 不使用 degree/connectivity protection。这是项目明确选择，不要擅自加回。
- 原论文所谓 representation modulation 不应被机械地改成“前20轮完全关闭、过滤后突然打开”。实验表明 cross norm 更像 backbone 稳定器。
- 当前推荐衔接：cross norm 从 epoch 1 始终启用；过滤后仅把无权 RMS 统计切换为 reliability-weighted RMS。
- 当前代码主对照：`original_always` 与 `reliability_weighted_always`。
- 当前代码严格采用用户指定的 direct cross norm；`lambda_` 在这些 NRGCF 模式中不参与 forward。
- 所有训练入口使用全局 Recall@20 early stopping：连续 20 个 epoch 没有严格提升即停止。
- `outputs_v1.6` 已完成：adaptive 规则在四组都于 epoch 7 触发；可靠性排序有效，但相对 CrossNorm 的快速峰值仍偏晚。
- `reliability_weighted_always` 在过滤切换当轮显著减小性能冲击，但下一轮很快被 `original_always` 追平，尚不能声称它稳定提高最终性能。
- 下一轮是最小 `outputs_v1.7` timing pilot：只跑 noise ratio `0.2`、seed `2026`，比较 no filtering、early adaptive + original、early adaptive + weighted。early adaptive 暂测 epoch 2–4、stable check 1；这不是已确定的最终超参数。
- `training_summary.json` 的 post-filter best 从 v1.7 起按 Recall@20 选择，并计入 filtering 触发当轮；旧结果漏掉了触发当轮。
- `outputs_v1.7` 已确认 early filter + original_always 相比 no-filter 提升 Recall 3.23%、NDCG 2.80%；weighted 略低于 original，不再作为主方法。
- 下一轮 `outputs_v1.8` 只补 clean no-filter、clean early fused filter，以及 20% noise 下同预算 `structure_weight=1.0` 排序消融。
- `outputs_v1.8` 已确认 clean early filtering 安全且有提升；20% noise 下 `structure_weight=1.0` 略优于 0.95，但单 seed 差距只有约 0.32% Recall。
- 下一轮 `outputs_v1.9` 仅用 seed 2027 复核 0.95 与 1.0 两种同预算排序，不重复 no-filter。
- 项目已决定固定 0.95 structure + 0.05 momentum，不再搜索融合权重。
- 项目明确保留推荐实验中常见的 test Recall@20 逐 epoch 评估、早停和 best epoch 选择，不新增 validation split；所有方法必须使用完全相同协议。
- 下一轮 `outputs_v2.0_amazon_book` 验证第二数据集：clean/20% noise × no-filter/95-5 early filtering，共四组。

## 必须避免的误读

- 原始日志的 `Recall@20=0.0679` 使用 `init_weight=0.01`、`lr=5e-4`，不能与默认 `init_weight=1`、`lr=1e-3` 的实验直接比较。
- 该原始日志最佳 epoch 是 19；如果只报告 overall best，可能完全没有使用 filtering 后的方法。
- 原始仓库的 filtering 只修改局部 `train_edge_index`，没有可靠地重建 `model.edge_index` 和训练 sampler。因此原始高分不是 edge denoising 有效性的证据。
- 曾经照搬论文的 `min(mean_norm_squared, 1)`，导致 divisor 几乎恒为 1，三个 modulation 模式完全相同。该实验无效，不得引用为 modulation 无效的证据。
- `origin/main` 是原始 NRGCF 仓库；`kbs/main` 才是本项目开发分支。服务器应从 NR-GCF-KBS- 仓库拉取。

## 2026-08-06 代码验收标准（历史）

- 三个传播层都要在 summary trace 中出现实际 RMS divisor。
- `original_always` 从 epoch 1 起 `effective_strength=1`。
- `reliability_weighted_always` 从 epoch 1 起也为 1；过滤前 RMS 与 original 相同，过滤后因 node confidence 加权而出现差异。
- 两个模式在过滤前必须得到相同 momentum、structure、删除预算和 removed edge IDs。
- 报告必须同时比较 overall best、best post-filter 和 final epoch，而不是只看 overall best。
