# 文档索引

> 命名维护（2026-09-22）：现行统一名称见登记表；SB旧组别A/B/C/D/E、RB旧称R、RD旧称v1/soft/P1/D1/E1按所属阶段映射，不将答案字母作方法名替换。见[方法版本登记](research/方法版本登记表.md)。

当前阶段（2026-10-06）：冻结工作基准仍为RD-1.2；Qwen32最好候选exp12为1710/2700（63.33%，对均匀+2.44个百分点），尚未达到该工作点3–5个百分点目标。新增LLaVA8验证：exp07/exp12为59.63%/59.41%，对均匀+3.67/+3.44个百分点，两版差值区间含0，见[八帧全量结果](experiments/video_mme_full/2026-10-05/llava8/实验结果.md)。此前exp12 Qwen16/LLaVA16为59.89%/61.63%，WFS公开选帧Qwen32为64.07%，见[跨配置验证](experiments/video_mme_full/2026-10-05/exp12_generalization/实验结果.md)。统一review后再讨论优化，阶段认识见[自主优化复盘](experiments/video_mme_200/2026-10-03/research_reset/阶段复盘.md)。

历史阶段（2026-09-22）：冻结工作基准为**RD-1.2（旧称RD-2P v1.0，检索密度引导的两遍选帧）**，开发200题125/200（62.5%）；BASE-Uniform/BASE-BLIP-TopK/RD-1.0/RD-1.1分别123/119/122/122题正确。取得小幅净收益，稳定优势尚待验证。统一ID及谱系见[方法版本登记](research/方法版本登记表.md)，规则与结果见[RD-1.2正式记录](experiments/video_mme_200/2026-09-22/density_e1/实验记录.md)。原设计、基线与旧实现全部保留。

| 分类 | 入口 | 内容 |
|---|---|---|
| exp27区域补查职责解耦 | [结果与职责复盘](experiments/video_mme_200/2026-10-08/region_scope/实验结果与职责复盘.md) · [协议](experiments/video_mme_200/2026-10-08/region_scope/执行协议.md) | 两配置开发136/126正确，与exp07持平；5400项离线选择，77新问答＋323输入核验导入，保留简化候选，无新全量成绩 |
| 单预算与内部设计收缩 | [审计结果与收缩建议](experiments/video_mme_full/2026-10-08/framework_contraction/审计结果与收缩建议.md) · [协议](experiments/video_mme_full/2026-10-08/framework_contraction/执行协议.md) | 新入口仅K，8100项选择重放一致；连接数可统一派生、连续触发不能直接替代；优先审查整合并区域继承补查，0模型调用 |
| 夜间入口与时间对照 | [ITC入口](experiments/video_mme_full/2026-10-07/itc_entry/定位验证.md) · [exp25时间目标](experiments/video_mme_full/2026-10-07/temporal_coverage/实验记录.md) · [exp26联合重复度](experiments/video_mme_full/2026-10-07/visual_temporal/执行与验证.md) | ITC可达68→56、时间目标丢船图均未问答；exp26开发130/200，对父新增5/退化13，不全量；目标继续、无新全量增益 |
| exp24独立视觉表征 | [竞争对照与停止结论](experiments/video_mme_full/2026-10-07/exp24_visual_state/实验记录.md) | 规范428真实＋4合成视觉前向、0QA；主对照1/3排序反转，物体状态有局部线索、船图保护风险，未进入开发/全量；调试成本单列 |
| BLIP空间状态诊断 | [实测与接入判断](experiments/video_mme_full/2026-10-06/coverage_objective_review/参考选帧与下一阶段决策.md#空间状态诊断实测2026-10-06) | 30真实BLIP配对＋4合成、0QA；接口对齐，3有效对照无独立排序增量，409对照标签错误保留，暂不接入 |
| 参考选帧与下一阶段决策 | [WFS对照与状态表征方案](experiments/video_mme_full/2026-10-06/coverage_objective_review/参考选帧与下一阶段决策.md) | 2700题双向得失、16题事后初审；035单区32名额仍遗漏逐行过程，优先输入层表征诊断，0新模型调用 |
| 覆盖目标复盘与下一步定位 | [定位与候选能力方案](experiments/video_mme_full/2026-10-06/coverage_objective_review/定位与候选方案.md) | 两组2700题、75事实、12题有界事后初审；统一权重不立项，局部证据状态信号待定，0新模型调用 |
| LLaVA八帧五策略全量 | [完整结果](experiments/video_mme_full/2026-10-05/llava8/实验结果.md) · [协议](experiments/video_mme_full/2026-10-05/llava8/执行协议.md) | exp07/12对均匀+3.67/+3.44个百分点；11932唯一问答＋1568共享结果、19合成，独立及完成恢复审计通过，两版差值区间含0 |
| exp07双模型16帧等权对照 | [完整结果](experiments/video_mme_full/2026-10-05/exp07_budget16/实验结果.md) · [协议](experiments/video_mme_full/2026-10-05/exp07_budget16/执行协议.md) | Qwen/LLaVA为60.33%/62.22%，对均匀+2.22/+2.07个百分点；对exp12小幅正增益但区间均含0，3806新问答＋1594导入验收完成 |
| exp12跨模型跨预算及参考验证 | [完整结果](experiments/video_mme_full/2026-10-05/exp12_generalization/实验结果.md) · [协议](experiments/video_mme_full/2026-10-05/exp12_generalization/执行协议.md) | 7900新问答＋200导入完成；16帧对均匀有正向点估计，WFS32较exp12高0.74个百分点、区间含0，不宣称领先 |
| exp12全量与有界复核 | [结果与定位分析](experiments/video_mme_full/2026-10-03/exp12/实验结果与定位分析.md) | Qwen32为1710/2700（63.33%），对均匀+2.44个百分点；对exp07净+18但区间含0，11题初审，不自动升级默认版本 |
| exp07全量与定位 | [全量结果和后续方向](experiments/video_mme_full/2026-09-30/exp07/实验结果与定位分析.md) | Qwen32为1692/2700（62.67%），对均匀+1.78个百分点；9题初审与局部前缀可达性诊断完成，未达3–5个百分点目标 |
| 开发200题优化候选 | [定位、推导与完整验证](experiments/video_mme_200/2026-09-28/rd12_refinement/定位与验证记录.md) | RD-1.2-exp07：Qwen32为136/200，对均匀130净+6（+3.0个百分点）；仅开发点结果，未证明全量或稳定优势 |
| Qwen32帧扩展 | [六组结果与三环节分析](experiments/video_mme_full/2026-09-24/qwen32/实验结果.md) · [协议](experiments/video_mme_full/2026-09-24/qwen32/执行协议.md) | 8100条完成；均匀/Top-K/RD为60.89%/61.11%/61.85%，24题有界初审与零问答旧池诊断完成 |
| Qwen全量与跨模型结果 | [Qwen三策略与六组对照](experiments/video_mme_full/2026-09-23/qwen_lmms/实验结果.md) | 均匀/Top-K/RD为58.11%/59.04%/59.89%；8100新答案验收通过，RD相对均匀净+48、相对Top-K净+23 |
| 当前阶段定位 | [2700题统计与窗口/补查/选择初审](experiments/video_mme_full/2026-09-23/stage_analysis/阶段定位分析.md) | 全量配对、18题有界初审、4处窗口外见证及反例；零新增模型调用，不直接调参 |
| 当前同协议三组结果 | [LMMs-Eval全量对照](experiments/video_mme_full/2026-09-23/lmms_methods/实验结果.md) | 均匀/Top-K/RD-1.2为60.15%/58.59%/60.59%；RD相对均匀净+12、区间含0，相对Top-K净+54 |
| 当前参考协议均匀基线 | [LMMs-Eval全量均匀结果](experiments/video_mme_full/2026-09-23/lmms_uniform/实验结果.md) | 1624/2700（60.15%）；后续同协议比较，不能与旧RD成绩直接相减 |
| 当前基线对齐 | [LMMs-Eval桥接结果](experiments/video_mme_full/2026-09-23/lmms_bridge/实验结果.md) | 900视频与20次诊断完成；同帧旧6/10、新7/10，仅004-3变化；不是全量成绩 |
| 全量RD-1.2与跨模型阶段 | [2026-09-23全量阶段结果](experiments/video_mme_full/2026-09-22/rd12_qwen/实验结果.md) | LLaVA 1592/2700（58.96%），均匀/Top-K 1572/1557；Qwen-FA2合成验收通过，后续全量暂缓，优先协议对齐与桥接 |
| 统一命名与谱系 | [方法版本登记](research/方法版本登记表.md) | BASE/SB/RB/RD、冻结版本、exp/diag、旧别名及运行命名规则 |
| RD-1.2-exp01 | [入口人工修复诊断](experiments/video_mme_200/2026-09-22/entry_repair/诊断记录.md) | 七题粗候选诊断，不是自动算法或新问答成绩 |
| RD-1.2-exp02 | [单帧推广资格探针](experiments/video_mme_200/2026-09-22/promotion_probe/验证记录.md) | 64次辅助分类，未通过硬剔除门槛，不纳入RD-1.2 |
| 当前冻结工作版本 | [RD-1.2结果与复盘](experiments/video_mme_200/2026-09-22/density_e1/实验记录.md) | 125/200；5852文件内容指纹；9道得失＋3道控制，后续优先验证泛化和区域有效性 |
| RD-1.1-exp03局部距离尺度 | [固定配额排序验证](experiments/video_mme_200/2026-09-22/local_distance_scale/验证记录.md) | 200题离线、12题复核；恢复船图但损失VACATION气泡，新增模型调用0 |
| RD-1.1-exp02预算递减归因 | [密度奖励递减验证](experiments/video_mme_200/2026-09-22/density_reward_decay/验证记录.md) | 200题离线，11题图板初审；减少集中但有信息得失，未新增问答 |
| RD-1.1-exp01离线归因 | [跨区评分验证](experiments/video_mme_200/2026-09-21/priority_decoupling/验证记录.md) | 200题只改跨区w×u；恢复作品却丢失船图，未新增问答，不作为默认替代 |
| 200题扩展评估 | [四版本扩展记录](experiments/video_mme_200/2026-09-20/density_extension/实验记录.md) | 800条齐全；新增150题RD-1.1与均匀持平，8例实际选帧复核及预算偏置推导 |
| RD-1.1单因素验证 | [固定四秒RD-1.1结果](experiments/video_mme_50/2026-09-20/density_soft/实验结果.md) | 50题56%；相对RD-1.0改善383-2、退化260-2/691-3；保留局部证据，不替换默认版本 |
| RD-1.0冻结设计稿 | [检索密度引导的条件补查与选帧](research/检索密度引导的条件补查与选帧框架.md) | 原文v1.0对应RD-1.0，字节纳入历史实验哈希；当前RD-1.2规则见其正式记录 |
| 新框架初版测试 | [RD-1.0结果](experiments/video_mme_50/2026-09-20/retrieval_density_v1/实验结果.md) | 50题58%；完整配对得失、补帧来源、实际图板初查、成本与恢复验证 |
| RB-1.0区域边界实验 | [RB-1.0结果与复盘](experiments/video_mme_50/2026-09-17/region_boundary_v1/实验结果.md) | 固定版本无准确率收益，保留可复算代码，不否定局部取证方向；新认识见研究计划 |
| 全量基线结果 | [完整2700题结果](experiments/video_mme_full/2026-09-16/实验结果.md) | 两组5400条、预算上限修订、全部尝试成本与数据集边界 |
| 全量代码review | [代码与结果复核](experiments/video_mme_full/2026-09-16/代码与结果复核_20260917.md) | 成绩验证通过；接口/校验/统计回归待修复，不改源运行 |
| 阶段设计复盘 | [阶段复盘与探索方向](experiments/video_mme_50/2026-09-16/阶段复盘与探索方向.md) | 逐帧证据、六方面设计反思、保留内容及候选方向 |
| 研究时间线 | [研究时间线](research/研究时间线.md) | 从初始直觉到逐帧证据，持续记录重要认识、反例和决策变化 |
| SB系列详解 | [SB方法论框架详解](research/核心方法论框架详解.md) | 原理、逐步实现、数字示例、SB-1.1/SB-1.3差异及代理判断边界 |
| SB-1.3／SB-1.3-exp01 | [局部保护与查询结果](experiments/video_mme_50/2026-09-15/method_followups/实验结果.md) | 50新问答、50范围分类、50查询；局部保护与查询分开记录 |
| 改进执行入口 | [方法改进验证入口](development/方法改进验证入口.md) | 后台执行、调用账本、恢复与统计代码切入点 |
| 当前证据审计 | [最终证据审计报告](experiments/video_mme_50/2026-09-15/evidence_audit/最终证据审计报告.md) | 50题双尺度初审、300组事实关联、128条逐组归因、16次诊断与待确认单因素方案 |
| 早期全量报告 | [全量初审与诊断结果](experiments/video_mme_50/2026-09-15/evidence_audit/全量初审与诊断结果.md) | 当时的初审版本；004-3的82秒阶段名称等解释以最终报告/r6更正为准 |
| 首批历史审计 | [首批初审与待复核](experiments/video_mme_50/2026-09-15/evidence_audit/首批初审与待复核.md) | 首批10题及当时未完成项，保留历史 |
| 审计工具 | [证据审计工具](development/证据审计工具.md) | 匿名复核、PTS视图、CSV和追加式复核导入 |
| 研究主计划 | [研究与实验计划](research/研究与实验计划.md) | 当前目标、实验约定、重要结果与下一步决策 |
| 待讨论事项 | [方法待验证问题](research/方法待验证问题.md) | 未确认修改、解释边界与实验注意事项 |
| 环境与推理 | [LLaVA推理与环境](setup/LLaVA推理与环境.md) | 专用Conda环境、官方源码与单题验收 |
| 代码与命令 | [代码结构与运行入口](development/代码结构与运行入口.md) | 模块职责、后台执行、续跑与统计 |
| 方法实现 | [首版实现说明](development/方法首版实现说明.md) | 中文接口与步骤说明、后台执行和审计 |
| 方法结果 | [首版结果与复盘](experiments/video_mme_50/2026-09-14/method_v1/实验结果.md) | 四组对照、案例证据与未执行的优化提案 |
| 实验进度 | [实验进展记录](experiments/实验进展记录.md) | 阶段进展和历史记录 |
| 实验索引 | [各轮实验](experiments/README.md) | 按数据集／日期查找协议与报告 |
| 实验模板 | [实验记录模板](experiments/实验记录模板.md) | 新一轮的计划和结果填写模板 |
| 相关工作 | [论文索引](related_work/README.md) | `analysis/`中文解析、`pdfs/`论文及来源哈希 |
| 历史资料 | [环境与资产核验（2026-09-14）](archive/环境与资产核验_20260914.md) | 迁入时的状态、历史缺项与原定revision，不作为当前环境结论 |

## 文档职责

- **研究与实验计划**：现行约定、重要结果摘要及后续决策，是当前状态的主入口。
- **实验进展记录**：按日期记录已执行工作、关键结论和待办，不重复配置及完整方法说明。
- **研究时间线**：保留认识变化、反例和更正理由，不重复完整实验统计。
- **单轮执行协议／结果报告**：维护该轮固定条件、版本修订、完整统计与成本；代码审查记录工程问题和复现证据。
- **方法详解／阶段复盘**：分别说明实现原理和当时的设计讨论，不将历史约定写成当前默认值。

新实验的协议和报告放在`experiments/<数据集>/<日期>/`；同日多轮增加子目录，不为每次状态更新新建报告。逐题结果、日志、模型和视频保留在`outputs/`和`data/`，不放入docs。本文链接相对所在文档；命令和代码路径以项目根目录为起点。

目录整理只调整当前维护文档。已完成运行中的 `protocol.json`、`code_snapshot/`、报告副本和结果包属于历史证据，保持原样。
