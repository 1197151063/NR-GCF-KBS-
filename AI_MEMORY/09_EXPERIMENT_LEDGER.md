# 当前实验账本（2026-08-14）

这是新 session 判断实验状态时的最高优先级文件。早期 `AI_MEMORY/04_EXPERIMENT_HISTORY.md` 保留方法演化过程，但其中“下一轮”与“当前推荐”已经过时。

## 权威留档

- 人类可读总表：[`../experiment_archive/2026-08-14/README.md`](../experiment_archive/2026-08-14/README.md)
- 机器可读结果：[`../experiment_archive/2026-08-14/results_snapshot.json`](../experiment_archive/2026-08-14/results_snapshot.json)
- 原始结果来源与哈希：[`../experiment_archive/2026-08-14/source_manifest.json`](../experiment_archive/2026-08-14/source_manifest.json)
- 当前 BPR 完整 profile：[`../configs/full_bpr_edge_filter_norm.json`](../configs/full_bpr_edge_filter_norm.json)
- 当前四臂运行器：[`../code/run_full_edge_filter_norm_bpr.sh`](../code/run_full_edge_filter_norm_bpr.sh)

## 当前方法状态

- 主方法是 structure--momentum hard edge filtering + epoch-1 always-on blended CrossNorm。
- Filter risk 以 bilateral two-hop structural inconsistency 为主、EMA BPR loss 为补充；删除预算由高-loss/低-structure consensus 得出，再应用 dataset-specific cap。
- CrossNorm 是 propagation operator，不是 filtering 后才启动的第二阶段模块。
- Full filtering 当前只针对 BPR。SSM/AU 只完成无过滤 CrossNorm compatibility，不得写成 full-method objective generalization。
- 不使用 degree/connectivity protection；但图会在删边后正确重建，sampler 与 propagation graph 保持一致。

## 当前 dataset profile

| Dataset | `mu` | `w_s` | Cap | Timing | LR | Decay |
|:---|---:|---:|---:|:---|---:|---:|
| Yelp2018 | 0.4 | 0.6 | uncapped consensus | adaptive 2--4 | 5e-4 | 1e-4 |
| Amazon-Book | 0.4 | 0.6 | uncapped consensus | adaptive 2--4 | 5e-4 | 1e-4 |
| LastFM | 0.2 | 0.95 | 4% | fixed 10 | 5e-4 | 1e-3 |
| ML-1M | 0.2 | 0.95 | 0.5% | adaptive 5--10 | 5e-4 | 1e-3 |

共同：500 epochs 上限，test Recall@20 patience 20，seed 2026，normal std 0.01，batch 2048，uniform degree-preserving replacement noise。

## 当前正在运行的主实验

使用同一代码版本和同一 shared dataset profile，在四个数据集上比较：

1. LightGCN；
2. Full Filter+Norm。

noise ratio 为 `0, 0.1, 0.2, 0.3, 0.4, 0.5`，seed 2026，共 48 runs。先得到完整 robustness curve；`Norm-only` 与 `Filter-only` 作为后续组成消融，不在本轮 48 runs 中重复。

## 必须避免的错误引用

- Yelp v3.3 SSM `+9.46%` 是错误 SSM 公式的结果；不能引用。
- Yelp v3.3 AU `+3.01%` 不依赖该 SSM 分支，可作为 exploratory evidence。
- 早期 LastFM/ML-1M 100-epoch baselines 未收敛，已被 500-epoch结果替代。
- `min(mean_norm_squared, 1)` 导致的相同 modulation 结果是实现退化。
- delayed/two-stage Norm 不是当前方法。
- 不得宣称 CrossNorm 对每个 dataset/objective 都提升；Amazon SSM/AU 与 corrected Yelp SSM 是负结果。
