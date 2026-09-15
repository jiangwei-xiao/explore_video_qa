#!/usr/bin/env bash
set -euo pipefail
# 步骤1：定位既有研究环境，新增方法依赖不覆盖基线的锁文件和代码。
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${VIDEOQA_METHOD_PYTHON:-/App/conda/envs/explore_video_qa/bin/python}"
cd "$PROJECT_DIR"
"$PYTHON" -c 'import torch, transformers; assert torch.__version__ == "2.6.0+cu124"; assert transformers.__version__ == "4.45.2"'
# 步骤2：以已核验的基础包版本作约束，安装单独锁定的方法与绘图依赖。
"$PYTHON" -m pip install -c requirements-method-constraints.txt -r requirements-method.lock.txt
"$PYTHON" -m pip check
# 步骤3：只取固定版本的分段参考用于测试，实际方法实现保存在独立方法包。
REF_DIR="$PROJECT_DIR/third_party/WFS-SB-reference"
mkdir -p "$REF_DIR"
if [[ ! -f "$REF_DIR/core.py" ]]; then
    curl -fLsS --connect-timeout 15 --max-time 60 \
      https://raw.githubusercontent.com/MAC-AutoML/WFS-SB/a424fc4528ecbe57edc93413826a2f2b8bb2c203/wfs/core.py \
      -o "$REF_DIR/core.py.part"
    printf '%s  %s\n' 86614d63415abea4d65578c65710d32f68e422206d9da9b80381e9f2812106f0 "$REF_DIR/core.py.part" | sha256sum -c -
    mv "$REF_DIR/core.py.part" "$REF_DIR/core.py"
fi
printf '%s  %s\n' 86614d63415abea4d65578c65710d32f68e422206d9da9b80381e9f2812106f0 "$REF_DIR/core.py" | sha256sum -c -
# 步骤4：检查机制与接口注释；GPU模型检查可按文档单独开启。
"$PYTHON" -m pytest -q
