# 实验索引

> 命名维护（2026-09-22）：现行统一名称见登记表；SB旧组别A/B/C/D/E、RB旧称R、RD旧称v1/soft/P1/D1/E1按所属阶段映射，不将答案字母作方法名替换。见[方法版本登记](../research/方法版本登记表.md)。

| 数据集／日期 | 执行协议 | 结果报告 | 原始运行编号 |
|---|---|---|---|
| RD-1.2／开发200题／2026-09-22 | [固定两遍流程](video_mme_200/2026-09-22/density_e1/实验记录.md) | [125/200与版本冻结](video_mme_200/2026-09-22/density_e1/实验记录.md) | `density_e1_200_20260922_r1` |
| RD-1.2-exp02／2026-09-22 | [单帧能力边界](video_mme_200/2026-09-22/promotion_probe/验证记录.md) | [64次辅助识别](video_mme_200/2026-09-22/promotion_probe/验证记录.md) | `promotion_probe64_20260922_r1` |
| RD-1.2-exp01／2026-09-22 | [七题粗候选诊断](video_mme_200/2026-09-22/entry_repair/诊断记录.md) | [人工入口修复](video_mme_200/2026-09-22/entry_repair/诊断记录.md) | `entry_repair7_20260922_r1` |
| RD-1.1-exp03／2026-09-22 | [固定配额局部排序](video_mme_200/2026-09-22/local_distance_scale/验证记录.md) | [离线结果，无问答](video_mme_200/2026-09-22/local_distance_scale/验证记录.md) | `local_distance_scale_200_20260922_r1` |
| RD-1.1-exp02／2026-09-22 | [密度奖励递减](video_mme_200/2026-09-22/density_reward_decay/验证记录.md) | [离线结果，无问答](video_mme_200/2026-09-22/density_reward_decay/验证记录.md) | `density_reward_decay_200_20260922_r1` |
| RD-1.1-exp01／2026-09-21 | [跨区评分解耦](video_mme_200/2026-09-21/priority_decoupling/验证记录.md) | [离线结果，无问答](video_mme_200/2026-09-21/priority_decoupling/验证记录.md) | `priority_decoupling_200_20260921_r1` |
| BASE两组／RD-1.0／RD-1.1／开发200题 | [扩展与来源](video_mme_200/2026-09-20/density_extension/实验记录.md) | [四方法结果](video_mme_200/2026-09-20/density_extension/实验记录.md) | `density_extension200_20260920_r1` |
| RD-1.1／开发50题／2026-09-20 | [执行协议](video_mme_50/2026-09-20/density_soft/执行协议.md) | [结果与固定输入复测](video_mme_50/2026-09-20/density_soft/实验结果.md) | `density_soft_videomme50_20260920_r1`；复测另列 |
| RD-1.0／开发50题／2026-09-20 | [执行协议](video_mme_50/2026-09-20/retrieval_density_v1/执行协议.md) | [初版结果](video_mme_50/2026-09-20/retrieval_density_v1/实验结果.md) | `retrieval_density_v1_videomme50_20260920_r1` |
| Video-MME区域边界取证RB-1.0／2026-09-17 | [执行协议](video_mme_50/2026-09-17/region_boundary_v1/执行协议.md) | [离线门槛与50题结果](video_mme_50/2026-09-17/region_boundary_v1/实验结果.md) | `region_boundary_v1_videomme50_20260917_r1` |
| Video-MME全量2700题／2026-09-16—17 | [初始协议与预算上限修订](video_mme_full/2026-09-16/执行协议.md) | [5400条结果](video_mme_full/2026-09-16/实验结果.md) · [代码review](video_mme_full/2026-09-16/代码与结果复核_20260917.md) | `videomme_full_uniform_topk_20260916_r1`＋`videomme_full_completion_20260917_r2`；失败补跑r1另计成本 |
| Video-MME SB-1.3与查询／2026-09-15 | [执行协议](video_mme_50/2026-09-15/method_followups/执行协议.md) | [50条新问答与查询离线结果](video_mme_50/2026-09-15/method_followups/实验结果.md) | `followups_clocal_query_videomme50_20260915_r1` |
| Video-MME证据审计／2026-09-15 | [审计与诊断协议](video_mme_50/2026-09-15/evidence_audit/全量审计与诊断协议.md) | [最终审计报告](video_mme_50/2026-09-15/evidence_audit/最终证据审计报告.md) | `evidence_videomme50_20260915_r1`；诊断`evidence_repairs_20260915_r1` |
| Video-MME开发50题／2026-09-14 | [均匀与Top-K协议](video_mme_50/2026-09-14/执行协议.md) | [100条基线结果](video_mme_50/2026-09-14/实验结果.md) | `videomme50_uniform_topk_20260914_r1` |
| Video-MME方法首版／2026-09-14 | [四组执行协议](video_mme_50/2026-09-14/method_v1/执行协议.md) | [200条结果与机制复盘](video_mme_50/2026-09-14/method_v1/实验结果.md) | `method_v1_videomme50_20260914_r1` |

基线与方法原始证据分别位于`outputs/baselines/`和`outputs/methods/`，审计与诊断位于`outputs/analysis/`和`outputs/diagnostics/`。各轮调用规模、补跑和成本由对应结果报告维护；旧单题验收不混入正式成绩，已完成运行及源码快照保持冻结。

阶段信息见[实验进展记录](实验进展记录.md)。新一轮使用[实验记录模板](实验记录模板.md)，先明确问题、条件和调用预算，再执行；运行方式见[代码与命令说明](../development/代码结构与运行入口.md)。
