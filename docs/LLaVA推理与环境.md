# LLaVA推理与环境

2026-09-14：基于本机共享权重与新拉取的官方代码完成重建。当前推理链路不依赖旧会话的脚本、环境或验收记录。

## 专用Conda环境

环境名 `explore_video_qa`，路径 `/App/conda/envs/explore_video_qa`。Python 3.10.21、PyTorch 2.6.0+cu124、Torchvision 0.21.0、Transformers 4.45.2、Tokenizers 0.20.3，FFmpeg／FFprobe 9.0.1可用。环境独立安装依赖，不继承 `conda_xmake` 的site-packages。

```bash
source /App/conda/etc/profile.d/conda.sh
conda activate explore_video_qa
cd /home/admin/projects/explore_video_qa
```

非交互SSH不依赖自动激活，显式使用 `/App/conda/envs/explore_video_qa/bin/python`。

重建入口为 `bash scripts/setup_llava.sh`；`environment.yml`描述Conda环境，`requirements-runtime.txt`列出推理依赖，`requirements-runtime.lock.txt`冻结本次pip版本，`conda-linux-64.lock.txt`冻结本次Conda平台包。复现精确Conda包时可用 `conda create -n <新环境名> --file conda-linux-64.lock.txt`，随后安装pip锁定文件和固定源码。共享模型路径仍需在目标机准备。

## 官方源码和本地模型

官方仓库：[LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)，固定commit `bce12e479bc4dfee2b9c50c88137b01ff51bd483`，源码位于 `third_party/LLaVA-NeXT/`，未修改官方文件。模型加载、视觉预处理、视频序列组装与生成调用均使用官方实现，依据[该版本官方视频示例](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/bce12e479bc4dfee2b9c50c88137b01ff51bd483/docs/LLaVA_Video_1003.md)。

`configs/llava_source_manifest.json`记录commit及源码哈希，`configs/local_model_inventory.json`记录LLaVA／SigLIP／BLIP本地推理文件的完整SHA256。它冻结本地字节身份，尚未比对上游LFS哈希。

SigLIP保留官方标识 `google/siglip-so400m-patch14-384`，通过 `.cache/huggingface/`快照链接只读使用 `/home/models` 权重。缓存键为本地内容／文件状态派生值，不冒充官方revision；`refs/main`没有末尾换行。推理启用离线模式，共享权重和原配置未改写。

## 已实现的入口

- `scripts/prepare_runtime.py`：固定源码与模型文件身份，建立离线缓存映射。
- `src/videoqa_runtime/video.py`：按源PTS进行1 FPS、0.25秒相位扫描，保存真实帧编号和时间；均匀组使用 `round(linspace(0,N−1,16))`，再按精确PTS解码选中的RGB帧。
- `src/videoqa_runtime/llava_backend.py`：官方加载与生成，BF16＋SDPA、`qwen_1_5`模板、贪心生成、最多8个新Token。输入16个唯一帧，分辨率384×384；实际prefill长度与视觉Token不符时停止。
- `scripts/run_llava_smoke.py`：对冻结开发集中的单题运行，生成前写入调用记录，保存原始输出、完整提示词、源帧、Token、耗时、显存及结果。已有输出路径拒绝覆盖，不按答题正确与否重试。
- `scripts/capture_runtime_acceptance.py`：从本次结果生成简要验收记录及环境锁。

时间网格映射规则为“取时间不早于目标时刻的第一张源帧”，按源帧去重；变帧率视频不使用平均FPS推算源时间。不同选择器共享中性的时间提示，不向Top-K声称帧是均匀抽取的。标准答案仅在生成后计分，不进入提示词或选择器。

示例（新输出路径；已完成的同题同协议结果应先检查是否可复用）：

```bash
python scripts/run_llava_smoke.py \
  --question-id 264-2 --gpu 0 \
  --output outputs/acceptance/new_run.json
```

当前尚未实现BLIP批量评分、Top-K调度、完整50×2断点续跑及研究方法模块；BLIP仅完成文件哈希，不能把这次LLaVA验收视为评分器运行验收。

## 本机实际验收

固定开发题 `264-2` 仅生成1次，输出 `A.`，可解析为A。16帧预处理形状为 `[16,3,384,384]`，实际prefill为3585 Token，其中视觉序列3360 Token。生成约0.98秒，峰值已分配显存约19.29 GiB；包含视频解码和模型加载的脚本耗时约11.13秒。这是单题运行验收，不是准确率估计或长视频平均速度。

记录：`outputs/acceptance/llava_264-2.json`（完整逐题证据），[model_runtime_validation.json](../configs/model_runtime_validation.json)（小型验收摘要）。15项协议／时间戳测试通过，`pip check`通过，运行后GPU已释放。

官方嵌套SigLIP加载产生448条meta-copy警告，完整保留在结果中。最终无meta参数；语言嵌入、投影、newline与两组视觉权重的确定切片共5项均与外层LLaVA checkpoint一致，实际生成各步下一Token logits有限。没有将这些切片检查宣称为内存中全部权重的逐元素核验。

官方导入时可能提示可选OpenCLIP／视频工具未安装。本项目使用PyAV解码；未安装可选Decord，因为其0.6.0发行包的内部wheel标签为CPython3.6，会被本环境的pip一致性检查判为不兼容。未修改官方源码或伪造依赖。

## 后续工作

旧会话资产现在只用于历史追溯与对照，不是重建前置条件。继续开发BLIP评分与两组基线时，先review当前输入协议和新入口，明确复用这一次验收结果的条件，再完成前5题两组验收及剩余调用。方法设计和冻结样本未改动，预留集未使用，每次Git提交前仍需review确认。
