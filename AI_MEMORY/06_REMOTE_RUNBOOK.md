# 远程 GPU 运行手册

## 服务器约定

- 代码目录：`/root/cyj/NR-GCF-KBS-/code`
- 输出根目录：`/root/autodl-tmp/outputs`
- 不写死其他用户名或服务器路径。
- 正式训练只在远程 GPU 运行。

## 更新代码

```bash
cd /root/cyj/NR-GCF-KBS-
git pull origin main
git rev-parse --short HEAD
```

拉取后应确认 `main` 包含全局 Recall@20 early stopping 和当前 always-on CrossNorm 实现；具体版本以本地最新提交为准。

注意：服务器 clone 的 remote 通常名为 `origin`，指向 NR-GCF-KBS-；本地开发机的该 remote 名为 `kbs`。

## 当前推荐实验：BPR 完整方法 noise curve

先确认本地新增的 profile、运行器和分析器已提交并推送，再在服务器执行：

```bash
conda activate cyj
cd /root/cyj/NR-GCF-KBS-/code

OUT=/root/autodl-tmp/outputs/outputs_v5.3_bpr_full_noise_curve_lr_split
mkdir -p "$OUT"

nohup env \
  GPU_ID=0 \
  OUTPUT_ROOT="$OUT" \
  DATASETS="yelp2018 amazon-book lastfm ml-1m" \
  NOISE_RATIOS="0 0.1 0.2 0.3 0.4 0.5" \
  SEEDS="2026" \
  ARMS="lightgcn full" \
  REQUIRE_CLEAN_REPO=1 \
  bash ./run_full_edge_filter_norm_bpr.sh \
  > "$OUT/run.log" 2>&1 &
```

查看进度：

```bash
tail -f /root/autodl-tmp/outputs/outputs_v5.3_bpr_full_noise_curve_lr_split/run.log
```

默认共 48 runs：四个数据集 × 六个 noise ratio × LightGCN/Full 两个方法。每个 case 使用独立目录；已完成 case 有 `comparison_summary.json` 时会跳过。若某个 case 中断，先只移动该 case 目录，再重新执行相同命令。脚本默认 `SUMMARY_ONLY=1`，不保留 per-edge CSV 和临时 generated train。

后续需要做严格组成消融时，可复用同一脚本并显式设置
`ARMS="lightgcn norm_only filter_only full"`。

完成后重点回传：

- `all_runs.json`
- `full_edge_filter_norm_summary.json`
- `full_edge_filter_norm_summary.md`
- `run.log`
- 各 case 的 `comparison_summary.json`
- 各 run 的 `edge_reliability/training_summary.json`、`reliability_summary.json`、`run_manifest.txt` 与 `training.log`

## 历史实验：Amazon-Book outputs_v2.0

```bash
cd /root/cyj/NR-GCF-KBS-/code

GPU_ID=0 \
OUTPUT_ROOT=/root/autodl-tmp/outputs/outputs_v2.0_amazon_book \
bash run_amazon_book_generalization_v2_0.sh
```

共四组：Amazon-Book 的 clean/20% replacement noise × no-filter/固定 95/5
early adaptive filtering。seed 2026，test Recall@20 early stopping，summary only。

## 上一轮权重确认实验：outputs_v1.9

```bash
cd /root/cyj/NR-GCF-KBS-/code

GPU_ID=0 \
OUTPUT_ROOT=/root/autodl-tmp/outputs/outputs_v1.9 \
bash run_momentum_ranking_confirmation_v1_9.sh
```

只跑 seed 2027、20% replacement noise 的两组同预算对照：
`structure_weight=0.95` 与 `1.0`。两个 arm 使用独立子目录，最终自动生成统一
`comparison_summary.json`。

## 上一轮 clean safety 实验：outputs_v1.8

```bash
cd /root/cyj/NR-GCF-KBS-/code

GPU_ID=0 \
OUTPUT_ROOT=/root/autodl-tmp/outputs/outputs_v1.8 \
bash run_clean_safety_and_momentum_ablation_v1_8.sh
```

只新增三个 run：clean no-filter、clean early fused filter、20% noise 下
`structure_weight=1.0` 的同预算排序消融。v1.7 的 noisy 0.95 fused run 不重复。

## 上一轮 early timing 实验：outputs_v1.7

只跑 20% replacement noise、seed 2026 的三组最小对照：

```bash
cd /root/cyj/NR-GCF-KBS-/code

GPU_ID=0 \
OUTPUT_ROOT=/root/autodl-tmp/outputs/outputs_v1.7 \
bash run_early_adaptive_filtering_v1_7.sh
```

三组分别为 no filtering + original always、early adaptive + original always、
early adaptive + reliability-weighted always。默认 min epoch 2、max epoch 4、
stable checks 1；不导出 per-edge CSV/Parquet。

## 上一轮 representation modulation 实验

一个 seed，clean 与 20% replacement noise，两种 always-on modulation：

```bash
cd /root/cyj/NR-GCF-KBS-/code

GPU_ID=0 \
NOISE_RATIOS="0 0.2" \
OUTPUT_ROOT=/root/autodl-tmp/outputs/outputs_v1.6 \
bash run_representation_modulation_ablation_100e.sh
```

脚本默认：

- dataset yelp2018；
- seed 2026；
- epochs 100；
- 全局 Recall@20 early-stopping patience 20；
- lr 0.0005；
- init weight 0.01；
- adaptive filtering：min epoch 5、max epoch 10、coverage 0.99、Jaccard 0.90、连续稳定检查 2 次；
- EMA decay 0.9；
- structure weight 0.95；
- summary only；
- 不保留大 CSV、临时 labels、generated train。

## 只跑 smoke

```bash
GPU_ID=0 \
NOISE_RATIOS="0.2" \
TRAIN_EPOCHS=30 \
OUTPUT_ROOT=/root/autodl-tmp/outputs/outputs_v1.5_smoke \
bash run_representation_modulation_ablation_100e.sh
```

## 检查 modulation trace

```bash
python - <<'PY'
import glob
import json
import os

root = "/root/autodl-tmp/outputs/outputs_v1.5"
for path in sorted(glob.glob(root + "/**/edge_reliability/training_summary.json", recursive=True)):
    data = json.load(open(path))
    modulation = data["representation_modulation"]
    print("\n", os.path.basename(os.path.dirname(os.path.dirname(path))))
    print("mode:", modulation["mode"])
    print("overall:", data["best_epoch"], data["best_recall_at_20"], data["best_ndcg_at_20"])
    print("post-filter:", data["best_post_filter_epoch"], data["best_post_filter_recall_at_20"], data["best_post_filter_ndcg_at_20"])
    for row in modulation["trace"]:
        if row["epoch"] in (1, 19, 20, 21, 25, 100):
            print(row)
PY
```

## 必须检查的公平性

`original_always` 与 `reliability_weighted_always` 在 epoch 1–20 应满足：

- loss trajectory 相同；
- recall/NDCG 相同；
- momentum statistics 相同；
- structure statistics 相同；
- removal budget 相同；
- removed count 与 IDs 相同；
- epoch 1–20 layer scales 相同。

过滤后才允许 RMS scale 和推荐结果不同。

## 环境检查

```bash
python --version
python - <<'PY'
import importlib.util
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("pyarrow installed:", importlib.util.find_spec("pyarrow") is not None)
PY
nvidia-smi
```

## 结果回传

至少拷回整个 compact output 目录，尤其：

- `comparison_summary.json`
- 每个 run 的 `training_summary.json`
- `reliability_summary.json`
- `schema.json`
- `training.log`
- `run_manifest.txt`
- `noise_generation.json`
- `noise_validation.json`

不要只复制截图或终端最后几行。
