# 实验索引

> 命名维护（2026-09-22）：现行统一名称见登记表；SB旧组别A/B/C/D/E、RB旧称R、RD旧称v1/soft/P1/D1/E1按所属阶段映射，不将答案字母作方法名替换。见[方法版本登记](../research/方法版本登记表.md)。

| 数据集／日期 | 执行协议 | 结果报告 | 原始运行编号 |
|---|---|---|---|
| RD-1.2-exp09／Qwen32开发200题／2026-10-01 | [五路完整验证协议](video_mme_200/2026-10-01/joint_seed_exp09/执行协议.md) | [137正确与开发候选边界](video_mme_200/2026-10-01/joint_seed_exp09/实验结果.md) | `rd12_exp09__qwen32__dev200__20261001_r1` |
| RD-1.2-exp08／Qwen32开发200题／2026-10-01 | [完整入口替换协议](video_mme_200/2026-10-01/option_seed_exp08/执行协议.md) | [133正确与固定案例](video_mme_200/2026-10-01/option_seed_exp08/实验结果.md) | `rd12_exp08__qwen32__dev200__20261001_r1` |
| RD-1.2-diag01／Qwen32开发200题／2026-10-01 | [入口独立对照协议](video_mme_200/2026-10-01/option_rank/执行协议.md) | [132正确及证据边界](video_mme_200/2026-10-01/option_rank/实验结果.md) | `option_rank_dev200_20261001_r1`；不改exp07 |
| 定位文本与四路检索探针／2026-10-01 | [九题协议](video_mme_full/2026-10-01/query_input_probe/执行协议.md) | [定位结果与边界](video_mme_full/2026-10-01/query_input_probe/结果与后续诊断.md) | `query_input_pilot9_20261001_r1`及两阶段逐选项控制 |
| exp07／Qwen32全量2700题／2026-09-30 | [固定协议](video_mme_full/2026-09-30/exp07/执行协议.md) | [1692正确与定位分析](video_mme_full/2026-09-30/exp07/实验结果与定位分析.md) | `rd12_exp07__qwen32__videomme2700__20260930_r1`；200导入＋2500新回答 |
| RD-1.2探索／Qwen32开发200题／2026-09-28 | [规则与逐阶段边界](video_mme_200/2026-09-28/rd12_refinement/定位与验证记录.md) | [exp07为136/200与全部分支记录](video_mme_200/2026-09-28/rd12_refinement/定位与验证记录.md) | `rd12_exp07__qwen__dev200__k32__20260928_r1`；此前对照各自保留 |
| Qwen32三策略／全量2700题／2026-09-24 | [固定配置与验收](video_mme_full/2026-09-24/qwen32/执行协议.md) | [8100条结果与三环节分析](video_mme_full/2026-09-24/qwen32/实验结果.md) | `uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01` |
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

2026-10-01新增：[RD-1.2-exp10严格恒定像素资格验证](video_mme_200/2026-10-01/constant_frame_exp10/执行协议.md) · [结果](video_mme_200/2026-10-01/constant_frame_exp10/实验结果.md)：exp07底座、开发200题Qwen32，135正确；空帧退出但对exp07净−1，不合入默认方法、不自动扩大全量。

2026-10-01新增：[局部证据组合离线验证](video_mme_200/2026-10-01/local_evidence_groups/离线验证记录.md)：exp07固定配额、200题零调用，最小缺失侧补齐未确认关系收益，不进入问答。

2026-10-02新增：[exp11帧组完成与diag02匹配控制](video_mme_200/2026-10-02/seed_bundles/执行协议.md) · [结果](video_mme_200/2026-10-02/seed_bundles/实验结果.md)：diag02为129/200、exp11为128/200，未达全量门槛，保留记录不合入。

2026-10-02新增：[exp12原问题加权覆盖预算](video_mme_200/2026-10-02/relevance_coverage/执行协议.md) · [结果](video_mme_200/2026-10-02/relevance_coverage/实验结果.md)：138/200，对exp07净+2，未达全量优先门槛，保留小幅正向候选。

2026-10-02新增：[exp13区域种子强度](video_mme_200/2026-10-02/seed_strength/执行协议.md) · [结果及局部前缀离线检查](video_mme_200/2026-10-02/seed_strength/实验结果.md)：138/200，与exp12持平，不开展全量；纯覆盖边际前缀留为零问答诊断。

2026-10-02新增：[exp14入口×预算交互结果](video_mme_200/2026-10-02/joint_relevance/实验结果.md)：137/200，与exp09持平，不能将两项开发收益相加；[exp15原核心锚点保护结果](video_mme_200/2026-10-02/core_anchor/实验结果.md)：138/200，与exp12持平，代理见证改善未带来净正确率增益。
