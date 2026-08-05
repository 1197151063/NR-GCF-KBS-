# 代码地图与真实行为

## 仓库与分支

- 原始仓库 remote：`origin`，指向 `1197151063/NRGCF.git`。
- KBS 开发 remote：`kbs`，指向 `1197151063/NR-GCF-KBS-.git`。
- 开发分支：`main`，应与 `kbs/main` 对齐。
- 当前记忆对应 commit：`b30a463`。

## 主要文件

- `code/NR-GCF.py`：训练主入口、warm-up、filter hook、图切换、训练与评估循环。
- `code/model.py`：LightGCN、NRGCF、cross norm、传播图重建。
- `code/dataloader.py`：数据读取、训练边、测试边和 sampler 所需数据。
- `code/procedure.py`：evaluation 实现。
- `code/parse.py`：命令行配置。
- `code/world.py`：解析后的全局配置与 device。
- `code/utils.py`：采样、日志、early stopping、辅助方法。
- `code/edge_diagnostics.py`：逐边 diagnostics 导出与二跳结构估计。
- `code/edge_reliability.py`：紧凑 reliability policy、stable EMA、summary。
- `code/generate_degree_preserving_replace.py`：degree-preserving edge swap noise。
- `code/run_edge_diagnostics_grid.sh`：远程隔离 worktree 运行器。
- `code/run_representation_modulation_ablation_100e.sh`：当前推荐 modulation 对照脚本。
- `code/summarize_reliability_runs.py`：合并 compact JSON。

## 当前 edge filtering 真实流程

对于 `hard_structure_momentum`：

1. 从 epoch 1 开始普通 BPR+L2 训练。
2. 每条原始训练边通过稳定 edge ID 更新 detached instance BPR EMA。
3. 在配置的 filtering epoch（当前推荐 T=20）冻结 EMA。
4. 在 filtering 前训练图上计算 deterministic two-hop MinHash structure。
5. 形成 structure-dominant fused risk。
6. 计算 adaptive removal budget。
7. 选出 fused risk 最高的固定数量边。
8. 同时更新：
   - `model.edge_index`；
   - `dataset.train_edge_index`；
   - `dataset.sampling_weights`；
   - BPR 正样本集合。
9. 新传播图通过 `gcn_norm` 重新归一化。
10. evaluation 始终屏蔽完整的过滤前 observed train split，保证不同方法候选集合一致。

## 原始代码与论文的重要差异

### Filtering

原始仓库代码：

- legacy momentum 更新公式在 epoch 超过 10 后可能产生负权重；
- 使用固定 beta=0.8，而论文描述自适应阈值；
- min-max 后最小值为 0，又使用 `score > 0`，因此最小值边也会被删除；
- 局部 `train_edge_index` 被过滤，但原始 `model.edge_index`、dataset sampler 未可靠同步。

KBS 分支已经为实验管线修复了图和 sampler 的同步，但没有伪装成论文原实现。

### Representation modulation

存在三种必须区分的语义：

1. 论文文字/公式：描述为第二阶段，公式出现 `min(mean_norm_squared, 1)`。
2. 原始 Git commit `53a290a`：从 epoch 1 开启，并使用
   `lambda*x_cross_norm + (1-lambda)*x`。
3. 用户明确指定的参考实现：从 epoch 1 开启，每层直接
   `x = cross_norm(x)`，不使用 lambda 混合。

当前 KBS 方法采用第 3 种 direct cross norm 作为基准。CLI 中保留 `lambda_` 只为兼容，它不参与当前 NRGCF direct 模式。

## 原始高分日志的解释限制

日志 `NR-GCF_yelp2018_20250807_081857.log`：

- `lr=5e-4`；
- `init_weight=0.01`；
- config 中显示 `lambda=0.6`；
- best epoch 19；
- Recall@20 0.0679；
- NDCG@20 0.0561。

该结果说明 always-on norm 和优化配置很重要，但不能证明过滤或第二阶段有效，因为：

- 最好点非常早；
- 原始 filtering 管线没有正确重建训练图；
- 日志显示 lambda，但用户给出的 direct 实现不会消费 lambda，具体运行版本必须以 commit/manifest 为准。
