# explore_video_qa

长视频问答中的固定预算选帧研究：问题范围引导的片段预算竞争与有限局部补查。

- [文档索引](docs/README.md)：研究、环境、实验与相关工作入口。
- [研究与实验计划](docs/research/研究与实验计划.md)：当前方法设计。
- [代码结构与运行入口](docs/development/代码结构与运行入口.md)：代码分工、后台运行和结果恢复。
- [最新基线结果](docs/experiments/video_mme_50/2026-09-14/实验结果.md)：开发50题两组100条结果、准确率和性能。
- [方法首版结果与复盘](docs/experiments/video_mme_50/2026-09-14/method_v1/实验结果.md)：四组消融/诊断、机制证据及下一轮建议。

远端工作目录为 `/home/admin/projects/explore_video_qa`。两组基线均为58%；方法首版A/B/C/D分别为56%/60%/56%/56%，200条问答与100次分类已完成。完整方法尚无净提升，优化提案未执行。执行环境为 `conda activate explore_video_qa`。
每次 Git 提交前必须 review 并确认。视频、模型、压缩包、缓存和下载PDF仅作本地资产，Git保留文档、清单与哈希。

## 目录约定

| 目录／文件 | 用途 |
|---|---|
| `src/videoqa_runtime/` | 可复用推理、采样、评分、调度和统计实现 |
| `scripts/` | 环境准备、单题验收、批量实验及汇总命令 |
| `tests/` | 输入协议、选帧规则、流式解码及恢复测试 |
| `docs/` | 分类维护的设计说明、执行协议、可读报告与相关工作 |
| `configs/` | 模型／源码身份及环境、验收记录 |
| `data/manifests/`、`data/metadata/` | 冻结样本和小型数据记录 |
| `data/videos/`、`data/archives/` | 本地视频和上传压缩包，Git忽略 |
| `outputs/` | 每次运行的原始结果、日志、协议与代码快照，Git忽略 |
| `third_party/`、`.cache/` | 固定版本第三方源码和可重建缓存，Git忽略 |
| `environment.yml`、`requirements-runtime*.txt`、`conda-linux-64.lock.txt` | 环境定义与锁定文件，保留在根目录供准备脚本使用 |
