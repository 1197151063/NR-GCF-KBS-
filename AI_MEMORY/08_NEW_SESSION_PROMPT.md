# 新 session 迁移提示词

将下面内容与 `AI_MEMORY/` 一起提供给新的 Codex session：

```text
请先完整阅读仓库根目录 AI_MEMORY/00_START_HERE.md，并按其中顺序读取
AI_MEMORY 下其余 Markdown。以这些文件和当前 Git HEAD 的真实代码为准，
不要依赖对 NR-GCF 论文的模糊记忆。

项目目标是为 NR-GCF 设计 structure-dominant edge reliability，并研究其与
原始 always-on cross norm 的自然衔接。禁止直接接入 NT-BPR/NT-SSM，禁止
复制其 contrastive objective 和四 alpha 设计，禁止使用 synthetic label
参与训练决策。

开始工作前请先：
1. 报告当前 HEAD、git status 和 remote；
2. 确认最新输出目录及 run manifest；
3. 区分已验证结果、失败实现和待验证假设；
4. 不在本地运行完整训练；
5. 不覆盖或重构无关代码。

当前最高优先级是审阅 outputs_v1.5，比较 original_always 与
reliability_weighted_always 的 overall best、best post-filter、final result、
过滤边集合和逐层 RMS trace。
```

## 新 session 首轮应回答的问题

- 当前代码 commit 是否至少为 `b30a463`？
- 最新输出是否真的使用该 commit？
- actual noise ratio 是否符合 requested ratio？
- 两个 always-on 模式过滤前是否严格一致？
- filtering 后 weighted RMS 是否与 unweighted RMS 不同？
- overall best 是否发生在 filtering 前？
- best post-filter 与 final result 谁更能支持方法结论？
- 是否有任何 synthetic label leakage？

## 迁移时不要只复制的内容

- 不要只复制最后一个 `comparison_summary.json`。
- 不要只复制训练日志最后的 BEST RESULT。
- 不要只提供论文公式而不提供 commit。
- 不要把 outputs_v1.0 当作 noise 0.2；它实际是 clean。
- 不要把 min-cap modulation 的相同结果当作有效消融。
