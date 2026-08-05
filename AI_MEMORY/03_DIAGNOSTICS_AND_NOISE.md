# Edge diagnostics、稳定 edge ID 与噪声协议

## Stable edge identity

- edge ID 是训练文件经 loader 展开的稳定列位置。
- 不使用 DataLoader shuffle 后的 batch position 作为永久 ID。
- sampler 返回的 index 用于把 instance loss 更新回原 edge ID。
- replacement noise generator 保持 edge 数量，输出顺序化 labels，验证 endpoint 与 edge ID 一一对应。
- 重复 user-item edge 会在 noise validation 阶段被拒绝。

## Stable EMA momentum

当前可信训练动态不是原始 legacy `model.update_momentum`，而是独立 `StableEdgeMomentum`：

`m_e <- decay*m_e + (1-decay)*loss_e`

- decay 默认 0.9；
- 第一次 observation 直接初始化为当前 loss；
- update 输入全部 detach；
- 不参与 backward；
- filtering 时要求所有训练边都至少观察一次；
- 推荐冻结时间 T=20。

## 结构分数

第一版采用训练图限定的二跳结构一致性：

- user-side：候选 item 与同一 user 的其他 item 在 user 邻域上的 degree-normalized overlap。
- item-side：候选 user 与同一 item 的其他 user 在 item 邻域上的 degree-normalized overlap。
- 两侧可用时取算术平均；只有一侧时使用该侧；缺失时为 NaN。
- 使用 deterministic bounded-neighbor MinHash 近似。
- 对目标 edge 的直接贡献做 leave-one-edge-out 解析处理/近似扣除。
- 不构造完整 item-item、user-user 或 node-node dense matrix。

## Reliability 与 fused risk

设：

- `s_rank`：structure 的 percentile rank，越高越可靠；
- `m_rank`：EMA momentum loss 的 percentile rank，越高越可疑。

当前：

`reliability = 0.95*s_rank + 0.05*(1-m_rank)`

`fused_risk = 1 - reliability`

删除预算：

`count(m_rank >= 0.8 AND s_rank <= 0.2)`

最终按 fused risk 降序删除预算数量的边，edge ID 升序处理 ties。

## Replacement noise

当前优先协议：`degree_preserving_replace`，uniform swap selection。

- 随机选择成对训练边并交换 item endpoint；
- user degree 不变；
- item degree 不变；
- 总边数不变；
- 拒绝 duplicate 和已有 edge；
- 20% ratio 表示被 replacement 的 edge positions 占原边数约 20%；
- synthetic label 仅在 policy 冻结后读取并用于评估。

项目决定暂不考虑复杂 hard replacement，因为真实噪声主要可抽象为误触，uniform replacement 已足够用于首篇验证。

## Compact outputs

当前正式 pilot 默认 `SUMMARY_ONLY=1`，避免传输巨大 CSV。

每个 run 至少保留：

- `edge_reliability/reliability_summary.json`
- `edge_reliability/training_summary.json`
- `edge_reliability/schema.json`
- `noise_generation.json`
- `noise_validation.json`
- `training.log`
- `run_manifest.txt`
- 根目录 `comparison_summary.json`

`training_summary.json` 当前还包括：

- overall best；
- best post-filter；
- final loss；
- propagation/BPR edge count；
- representation modulation mode；
- 每 epoch modulation trace；
- 每层 user/item RMS divisor。

## 数据泄漏约束

- structure、degree、EMA、node confidence 只使用 train graph。
- validation/test edge 不参与特征。
- synthetic label 不参与 feature、budget、mask、node confidence。
- evaluation mask 使用过滤前完整 train set，但这只是排除已观察物品，不进入训练特征。
