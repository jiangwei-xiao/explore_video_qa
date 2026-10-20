"""CPU预提取入口：只用已冻结公开元数据，不改变正式资格或研究协议。"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：对完整SHA包独立提取，原协调器可复用，不必停止下载或模型流水线。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--asset-root',type=Path,default=ROOT/'data/videos/longvideo_reason')
    parser.add_argument('--state-root',type=Path,required=True)
    args=parser.parse_args()
    from videoqa_set_value.longvideo_trial import metadata
    from videoqa_set_value.longvideo_preextract import preextract
    # 步骤1：只读取公开投影的源ID；抽样和计分仍由原协调器单独执行。
    public=metadata(args.asset_root)
    ids={r['video_id'] for r in public['rows']}
    # 步骤2：独立输出原包来源/成员SHA，绝不提前发布训练验证清单。
    print(json.dumps(preextract(args.asset_root,args.state_root,ids),ensure_ascii=False))


if __name__=='__main__':main()
