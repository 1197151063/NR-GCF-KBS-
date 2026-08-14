# NR-GCF-KBS 实验留档（2026-08-14）

本目录冻结截至 2026-08-14 的**有效聚合结果、当前超参数选择和下一轮完整实验协议**。原始日志与旧输出不在仓库内重复复制；`source_manifest.json` 记录了来源路径、SHA-256 和可引用范围，`results_snapshot.json` 保存机器可读数值。

## 1. 证据边界

- 现有结果大多为 `seed=2026` 的探索性实验，且采用项目既定的 test Recall@20 逐 epoch 选择与 patience 20 早停协议。适合确定方法和缩小参数范围，不应伪装成多随机种子最终表。
- noisy case 指 uniform degree-preserving replacement，`noise_ratio=0.2` 表示替换 20% 训练边；synthetic label 只用于运行后评价，不参与过滤特征或训练。
- 完整 `edge filter + CrossNorm` 目前仅对 BPR 定义并启动。SSM/AU 表格是无过滤条件下的 CrossNorm objective compatibility 证据。
- Yelp 的 fusion 与 modulation 是分阶段搜索：clean modulation 表建立在 clean 最优 `w_s=0.95` 上，noisy modulation 表建立在 noisy 最优 `w_s=0.6` 上。因此 `w_s=0.6, mu=0.4` 的共享 clean 组合仍需由本轮完整四臂实验补齐。

## 2. 当前完整模块配置

共同配置：BPR，500 epochs 上限，patience 20，batch size 2048，normal initialization (`std=0.01`)，EMA decay 0.9，momentum quantile 0.8，structure quantile 0.2，MinHash two-hop structure，top-k 10，chunk size 8192。由于普通 LightGCN 收敛明显更慢，无 CrossNorm arm 使用 `lr=1e-3`；带 CrossNorm arm 使用 `lr=5e-4`。

| Dataset | LightGCN LR | Full LR | Decay | Norm weight $\mu$ | Structure weight $w_s$ | Removal cap | Filter timing |
|:---|---:|---:|---:|---:|---:|:---|:---|
| Yelp2018 | 1e-3 | 5e-4 | 1e-4 | 0.4 | 0.6 | uncapped consensus budget | adaptive epoch 2--4, stable checks 1 |
| Amazon-Book | 1e-3 | 5e-4 | 1e-4 | 0.4 | 0.6 | uncapped consensus budget | adaptive epoch 2--4, stable checks 1 |
| LastFM | 1e-3 | 5e-4 | 1e-3 | 0.2 | 0.95 | 4% | fixed epoch 10 |
| ML-1M | 1e-3 | 5e-4 | 1e-3 | 0.2 | 0.95 | 0.5% | adaptive epoch 5--10, stable checks 2 |

`removal cap=1.0` 的真实语义是“不额外截断 consensus-derived budget”，不是删除全部边。

## 3. Filter 与 Norm 的已有 BPR 结果

### 3.1 Yelp2018

Fusion sensitivity：

| Noise | 最优 $w_s$ | Recall@20 | NDCG@20 | Removed |
|---:|---:|---:|---:|---:|
| 0 | 0.95 | 0.066854 | 0.054743 | 8.91% |
| 0.2 | 0.60 | 0.057129 | 0.046194 | 11.31% |

在各 noise case 的上述 fusion 最优点上继续搜索 Norm：

| Noise | 最优 $\mu$ | Recall@20 | NDCG@20 | Removed |
|---:|---:|---:|---:|---:|
| 0 | 0.60 | 0.068542 | 0.056604 | 8.48% |
| 0.2 | 0.40 | 0.059227 | 0.047874 | 11.35% |

最终共享 profile 选择 `w_s=0.6, mu=0.4`：它优先保留 noisy optimum，并避免按 clean/noisy 标签切换方法参数。该共享组合的 clean 精确结果尚待新四臂实验确认。

### 3.2 Amazon-Book

| Noise | $w_s$ | $\mu$ | Recall@20 | NDCG@20 | Removed |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.60 | 0.40 | 0.047039 | 0.036912 | 8.91% |
| 0.2 | 0.60 | 0.40 | 0.035649 | 0.027635 | 9.88% |

Amazon 的 clean/noisy 搜索都选择相同的 `w_s=0.6, mu=0.4`，因此该参数组合目前最稳定。

### 3.3 LastFM

无过滤 common Norm 搜索选择 `mu=0.2`：clean `0.312584/0.302001`，noise 0.2 `0.253798/0.233357`。固定 epoch 10、cap 4%、`w_s=0.95` 的完整过滤结果为：

| Noise | Norm-only Recall/NDCG | Filter+Norm Recall/NDCG | 说明 |
|---:|:---|:---|:---|
| 0 | 0.312584 / 0.302001 | 0.311659 / 0.300003 | clean 轻微下降 |
| 0.2 | 0.253798 / 0.233357 | 0.261312 / 0.238575 | noisy Recall 相对提高约 2.96% |

在 noise 0.2 下，4% 删边的 noisy removal rate 为 7.88%，removed precision 为 39.41%。结构排序并不等同于最高 precision：`w_s=0.5` precision 更高，但 `w_s=0.95` 最终 Recall 更好。

### 3.4 ML-1M

无过滤 BPR Norm 搜索选择 `mu=0.2`。在 cap sensitivity 中，`w_s=0.95, cap=0.5%` 是最保守且 noisy 有收益的选择：

| Noise | Norm-only Recall/NDCG | Filter+Norm Recall/NDCG | 说明 |
|---:|:---|:---|:---|
| 0 | 0.272258 / 0.254619 | 0.270421 / 0.254019 | clean Recall 下降 0.001837 |
| 0.2 | 0.210394 / 0.185835 | 0.211769 / 0.186149 | noisy Recall 提高 0.001375 |

0.2 noise 下 removed precision 为 48.96%。由于收益很小且 clean 有代价，ML-1M 必须保留 `LightGCN / Norm-only / Filter-only / Full` 四臂结果，不能只报 Full。

## 4. CrossNorm 的 objective compatibility

以下均为 clean graph、无 edge filtering；数值为 Recall@20。

| Dataset | Objective | LightGCN | 最优 CrossNorm | $\mu$ | Relative gain | 结论 |
|:---|:---|---:|---:|---:|---:|:---|
| Yelp2018 | AU | 0.070526 | 0.072650 | direct always-on | +3.01% | 正向 |
| Yelp2018 | corrected SSM | 0.073039 | 约 0.071890 | searched | negative | 不支持普遍增益 |
| Amazon-Book | SSM | 0.059870 | 0.052695 | 0.8 | -11.98% | 负向 |
| Amazon-Book | AU | 0.056484 | 0.055392 | 0.2 | -1.93% | 负向 |
| LastFM |
 | 0.211931 | 0.220630 | 0.2 | +4.10% | 正向 |
| ML-1M | BPR | 0.251381 | 0.272352 | 0.2 | +8.34% | 正向 |
| ML-1M | SSM | 0.221819 | 0.258735 | 0.2 | +16.64% | 正向 |
| ML-1M | AU | 0.211679 | 0.215553 | 1.0 | +1.83% | 正向 |

因此目前能写的是“CrossNorm 与多种目标兼容，并在多个数据集/目标上有效”，不能写成“对所有数据集和所有目标均提升”。

## 5. Objective-specific 超参数记录

- BPR full module：使用第 2 节的 dataset profile。
- Yelp corrected SSM 的 clean/noisy robust search：`lr=1e-4, tau=0.14, decay=1e-4`；即使调参后，CrossNorm 仍未超过该数据集的 tuned LightGCN-SSM。
- LastFM SSM：`tau=0.5`；ML-1M SSM：`tau=0.1`。当前 Norm sensitivity 对两者统一使用 `lr=5e-4, decay=1e-3`。
- LastFM AU：uniformity weight `0.1`，选 `mu=0.2`；ML-1M AU：uniformity weight `0.5`，选 `mu=1.0`。
- Amazon-Book objective pilot：SSM `tau=0.1`，AU uniformity weight `5`；两者 CrossNorm 均未带来提升。

## 6. 不得用于论文主结果的历史数据

- Yelp v3.3 中 SSM `+9.46%` 使用了后来确认有误的 SSM 实现；同目录的 AU 分支不受该 SSM 公式错误影响。
- delayed/two-stage Norm 低于 epoch-1 always-on Norm，已退出当前方法。
- 使用 `min(mean_norm_squared, 1)` 的 modulation 发生恒等退化，导致多个 arm 完全相同；该结果无效。
- 早期 LastFM/ML-1M 100-epoch LightGCN objective baseline 未充分收敛，已被 500-epoch + patience 20 结果取代。
- Yelp `clean_noisy_comparison` 与早期 factorial 只记录调试阶段配置，不能作为最终 SSM 对比。
- 中断、重复目录拒绝覆盖以及 `all_runs.json` 中 run count 为 0 的任务不构成实验结果。
- Adap-tau 是对比方法，不是本方法的 CrossNorm 组成部分。

## 7. 下一轮完整实验

脚本 `code/run_full_edge_filter_norm_bpr.sh` 当前默认比较两个相互独立、可续跑的 arm：

1. `lightgcn`：无 Filter、无 Norm；
2. `full`：Filter + Norm。

默认运行四个数据集、noise `0/0.1/0.2/0.3/0.4/0.5`、seed 2026，共 48 runs。只保留 compact JSON、manifest 和训练日志，不保留大规模 per-edge CSV。完成后自动生成 `all_runs.json`、`full_edge_filter_norm_summary.json` 和 Markdown 总表。`norm_only` 与 `filter_only` 保留为后续消融开关。

这轮先回答完整方法相对 LightGCN 的 clean 性能与 0.1--0.5 noise robustness curve。Norm 和 Filter 的独立主效应随后通过四臂消融回答。
