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

## 当前推荐实验

一个 seed，clean 与 20% replacement noise，两种 always-on modulation：

```bash
cd /root/cyj/NR-GCF-KBS-/code

GPU_ID=0 \
NOISE_RATIOS="0 0.2" \
OUTPUT_ROOT=/root/autodl-tmp/outputs/outputs_v1.5 \
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
