# 相关工作

2026-09-14：依据现有中文解析提供的arXiv链接下载PDF，文件名保持原解析中的相对链接不变。目录沿用现有 `releated_work` 拼写。

| 工作 | 中文解析 | 本地PDF | 实际下载版本 |
|---|---|---|---|
| AKS | [解析](analyse/01_AKS_Adaptive_Keyframe_Sampling.md) | [PDF](pdfs/AKS_CVPR2025.pdf) | [2502.21271v1](https://arxiv.org/abs/2502.21271v1) |
| MDP³ | [解析](analyse/02_MDP3_List-wise_Frame_Selection.md) | [PDF](pdfs/MDP3_ICCV2025.pdf) | [2501.02885v2](https://arxiv.org/abs/2501.02885v2) |
| A.I.R. | [解析](analyse/03_AIR_Adaptive_Iterative_Reasoning.md) | [PDF](pdfs/AIR_ICLR2026.pdf) | [2510.04428v3](https://arxiv.org/abs/2510.04428v3) |
| EFS | [解析](analyse/04_EFS_Event_Anchored_Frame_Selection.md) | [PDF](pdfs/EFS_2026.pdf) | [2603.00983v1](https://arxiv.org/abs/2603.00983v1) |
| WFS-SB | [解析](analyse/05_WFS-SB_Wavelet_Semantic_Boundary.md) | [PDF](pdfs/WFS-SB_CVPR2026.pdf) | [2603.00512v2](https://arxiv.org/abs/2603.00512v2) |

[sources.json](pdfs/sources.json)记录来源URL、PDF首页版本、页数、大小及SHA256。PDF通过结构解析与arXiv编号核对；这不表示既有解析中的全部数字已重新逐表复核。

解析中对“我们”的固定四段／20秒窗口比较属于早期方案，当前方法见[研究与实验计划](../研究与实验计划.md)。当前方案借用WFS-SB独立分段，研究问题范围引导的逐帧预算竞争和有限补查，未取得方法实验结论。历史解析保留，后续正式撰写论文时应基于当前方法逐项更新比较与数值来源。
