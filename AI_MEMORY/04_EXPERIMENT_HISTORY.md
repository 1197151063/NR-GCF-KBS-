# 关键实验历史与结论

以下结果均应结合各自 commit、配置和输出目录理解。不要跨配置直接比较。

## 结构信号与旧 momentum

在 Yelp2018、20% uniform replacement noise 的诊断中：

- structure AUROC：约 0.8923；
- structure AUPRC：约 0.6915；
- 原 legacy momentum AUROC：约 0.6584；
- 原 legacy momentum AUPRC：约 0.3843。

结论：结构信号非常强；原 momentum 实现有明显问题。

## outputs_v0.7 / outputs_v0.8：主要 filtering 基线

20% noise，seed 2026：

| 方法 | Recall@20 | NDCG@20 | 备注 |
|---|---:|---:|---|
| none | 0.044131 | 0.036344 | 无过滤 |
| current legacy | 0.044399 | 0.036394 | 仅删除 12 条，基本失效 |
| hard consensus | 0.045695 | 0.037408 | 删除 106,679 |
| hard structure-only | 0.045859 | 0.037372 | 同数量结构排序 |
| old global soft | 0.042267 | 0.034760 | 明显变差 |
| gated soft | 0.044335 | 0.036471 | 没有稳定超过 hard |

hard consensus：

- noisy removal rate：33.98%；
- clean removal rate：2.28%；
- removed precision：78.82%。

hard structure-only：

- noisy removal rate：34.92%；
- clean removal rate：2.05%；
- removed precision：81.00%。

结论：hard structure 比 consensus 稍好；全局 soft reliability 不适合当前实现。

## outputs_v0.9：clean structure-only

- removed：90,403（7.31%）；
- Recall@20：0.057611；
- NDCG@20：0.047241。

clean none 参考：

- Recall@20：0.057263；
- NDCG@20：0.047157。

说明结构 hard filtering 没有明显破坏 clean 性能。

## outputs_v1.0：错误 noise 配置但有 clean 参考价值

该轮本应测试 noise 0.2，但实际继承成 noise 0。不能当作 noisy 实验。

| T | 删除数 | Recall@20 | NDCG@20 |
|---:|---:|---:|---:|
| 10 | 83,029 | 0.057726 | 0.047398 |
| 15 | 91,071 | 0.057483 | 0.047216 |
| 20 | 96,993 | 0.057528 | 0.047383 |

T20 相比 clean none 约提升 Recall 0.46%、NDCG 0.48%。

## outputs_v1.1：stable EMA + structure-dominant fused hard filtering

Yelp2018，20% replacement noise，seed 2026，总 100 epochs：

| T | 删除比例 | Noisy removal | Clean removal | Precision | Recall@20 | NDCG@20 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 8.55% | 35.23% | 1.88% | 82.38% | 0.046246 | 0.037572 |
| 15 | 9.64% | 39.35% | 2.21% | 81.63% | 0.046493 | 0.037736 |
| 20 | 10.19% | 41.37% | 2.39% | 81.20% | 0.046610 | 0.037817 |

T=20 相比 none：

- Recall 提升约 5.62%；
- NDCG 提升约 4.05%。

T=20 相比 hard structure-only：

- Recall 提升约 1.64%；
- NDCG 提升约 1.19%。

稳定 EMA 的分类质量随时间提高：

| T | Momentum AUROC | Momentum AUPRC | Fused AUROC | Fused AUPRC |
|---:|---:|---:|---:|---:|
| 10 | 0.7174 | 0.4289 | 0.8942 | 0.7146 |
| 15 | 0.7579 | 0.4941 | 0.8946 | 0.7165 |
| 20 | 0.7834 | 0.5294 | 0.8949 | 0.7174 |

单独 structure：AUROC 0.8923、AUPRC 0.6915。融合后 AUPRC 达 0.7174，支持 momentum 提供互补排序信息。

当时结论：旧配置固定 T=20，不再搜索 10/15/20。该结论后来被 always-on CrossNorm 的快速收敛观察取代；当前方法改用 training-only adaptive timing，不能把这条历史结论当作现行配置。

## Representation modulation 失败路径

### 论文 min cap 版本

曾实现：

`sqrt(min(mean_norm_squared, 1) + eps)`。

当 mean norm squared 大于 1 时 divisor 几乎恒为 1，导致：

- no modulation；
- paper stage-two；
- reliability-weighted stage-two；

三者结果完全一样。该结果是实现退化，不是科学结论，禁止引用。

### Delayed stage-two direct norm

随后改成过滤前关闭 norm、过滤后直接开启原始 cross norm。用户观察该版本达不到原始 always-on norm 的高分。

结论：cross norm 是 backbone calibration，不应为了匹配论文叙事而在 warm-up 阶段关闭。

## 原始 always-on 日志

用户提供的原始 Yelp2018 日志：

- best epoch 19；
- Recall@20 0.0679；
- NDCG@20 0.0561；
- lr 0.0005；
- init_weight 0.01；
- config lambda 0.6；
- epoch 30 Recall/NDCG 已降至 0.0667/0.0553。

该结果推动当前设计改为 always-on cross norm，但它不能证明第二阶段有效。

## 下一轮 pending

预计目录：`outputs_v1.5`。

- dataset：Yelp2018；
- noise ratio：0、0.2；
- seed：2026；
- 当时使用 T=20；当前实现已改为 adaptive epoch 5–10；
- epochs=100；
- lr=5e-4；
- init_weight=0.01；
- 对照：`original_always`、`reliability_weighted_always`。
