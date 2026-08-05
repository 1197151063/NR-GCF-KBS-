# 研究目标、边界与禁止事项

## 总目标

研究一条 user-item interaction 的可靠性是否不应只由该边自身训练 loss 决定，还应结合：

- 用户侧邻域结构一致性；
- 物品侧邻域结构一致性；
- 多跳协同传播结构；
- 训练动态；
- 必要时的表示尺度校准。

目标不是把已有方法拼接到 NR-GCF，而是形成 NR-GCF 自己的噪声可靠性建模和鲁棒表示学习方法。

## 当前研究叙事

1. 训练早期的 per-edge EMA loss 描述优化难度和 memorization dynamics。
2. 二跳 user/item-side structure 描述边是否符合协同邻域。
3. 结构信号是主体，momentum 是小比例互补信号。
4. hard filtering 删除高风险边，同时更新 BPR 正样本集合和 LightGCN 传播图。
5. cross norm 是基础 encoder calibration，应从训练开始启用。
6. 过滤后剩余边的 reliability 可继续用于稳健估计 cross norm 的全局尺度，实现 filtering 与 representation modulation 的自然连接。

## 严格禁止事项

- 不直接接入 NT-BPR。
- 不直接接入 NT-SSM。
- 不复制对方的双向 contrastive objective。
- 不复制 UU、II、UI、IU 四个 alpha 机制。
- 不把两个项目简单串联。
- 不重新实现近似 NT-SSM。
- 不显式构造 dense node-by-node structural matrix。
- 不显式枚举所有 multi-hop neighbor pairs。
- synthetic noise label 只能用于决策后的评估，不能参与特征、过滤、权重或调参。
- 不使用 validation/test edge 计算 diagnostics 特征。
- 不为了连通性强行增加 degree protection；项目当前明确选择不使用该约束。
- 不在本地运行完整训练，不下载数据集，不假设本地有 CUDA。
- 不在没有匹配配置时比较方法结果。

## 实验原则

- 首先做单 seed、单数据集、少量有解释力的对照，不做无意义的大网格。
- replacement noise 是当前主要协议，因为它更接近误触导致的错误交互，同时保持边数与两侧 degree。
- 先确认方法能 work，再扩展 noise ratio、seed 和 dataset。
- 每次方法对照必须保证第一阶段、随机种子、过滤边集合或删除预算等关键变量可解释。
- 结果必须区分：整体最好、过滤后最好、最终 epoch。
- 一个方法若整体最好出现在 filtering 前，不能声称第二阶段带来了该最好结果。

## 论文定位

目标期刊是 KBS。要达到投稿要求，至少需要：

- 清晰且不是简单拼接的统一方法动机；
- 结构信号和 momentum 的互补证据；
- 与无过滤、原过滤、structure-only、momentum-structure filtering 的消融；
- representation modulation 独立贡献和与 reliability 衔接的证据；
- 多数据集、多 noise ratio 的一致趋势；
- clean 性能安全性；
- 复杂度和可扩展性说明。
