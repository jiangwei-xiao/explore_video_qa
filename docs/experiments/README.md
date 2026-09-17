# 实验索引

| 数据集／日期 | 执行协议 | 结果报告 | 原始运行编号 |
|---|---|---|---|
| Video-MME区域边界取证R／2026-09-17 | [执行协议](video_mme_50/2026-09-17/region_boundary_v1/执行协议.md) | [离线门槛与50题结果](video_mme_50/2026-09-17/region_boundary_v1/实验结果.md) | `region_boundary_v1_videomme50_20260917_r1` |
| Video-MME全量2700题／2026-09-16—17 | [初始协议与预算上限修订](video_mme_full/2026-09-16/执行协议.md) | [5400条结果](video_mme_full/2026-09-16/实验结果.md) · [代码review](video_mme_full/2026-09-16/代码与结果复核_20260917.md) | `videomme_full_uniform_topk_20260916_r1`＋`videomme_full_completion_20260917_r2`；失败补跑r1另计成本 |
| Video-MME C-local与查询／2026-09-15 | [执行协议](video_mme_50/2026-09-15/method_followups/执行协议.md) | [50条新问答与查询离线结果](video_mme_50/2026-09-15/method_followups/实验结果.md) | `followups_clocal_query_videomme50_20260915_r1` |
| Video-MME证据审计／2026-09-15 | [审计与诊断协议](video_mme_50/2026-09-15/evidence_audit/全量审计与诊断协议.md) | [最终审计报告](video_mme_50/2026-09-15/evidence_audit/最终证据审计报告.md) | `evidence_videomme50_20260915_r1`；诊断`evidence_repairs_20260915_r1` |
| Video-MME开发50题／2026-09-14 | [均匀与Top-K协议](video_mme_50/2026-09-14/执行协议.md) | [100条基线结果](video_mme_50/2026-09-14/实验结果.md) | `videomme50_uniform_topk_20260914_r1` |
| Video-MME方法首版／2026-09-14 | [四组执行协议](video_mme_50/2026-09-14/method_v1/执行协议.md) | [200条结果与机制复盘](video_mme_50/2026-09-14/method_v1/实验结果.md) | `method_v1_videomme50_20260914_r1` |

基线与方法原始证据分别位于`outputs/baselines/`和`outputs/methods/`，审计与诊断位于`outputs/analysis/`和`outputs/diagnostics/`。各轮调用规模、补跑和成本由对应结果报告维护；旧单题验收不混入正式成绩，已完成运行及源码快照保持冻结。

阶段信息见[实验进展记录](实验进展记录.md)。新一轮使用[实验记录模板](实验记录模板.md)，先明确问题、条件和调用预算，再执行；运行方式见[代码与命令说明](../development/代码结构与运行入口.md)。
