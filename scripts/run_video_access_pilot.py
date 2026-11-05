#!/usr/bin/env python3
"""接口：受控原片访问试点入口，支持阶段执行、后台协调和安全恢复。"""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')


def main():
    """接口：所有GPU子阶段显式绑定单卡；完成恢复先核验已有记录再决定是否加载模型。"""
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['engineering','coordinate','select','inputs','qa','replay','report','audit'])
    p.add_argument('--root',required=True);p.add_argument('--dataset',choices=['videomme','mlvu'],default='videomme')
    p.add_argument('--gpu');p.add_argument('--slot',type=int,default=0);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--gpus',default='4,5,6,7');a=p.parse_args();root=Path(a.root).resolve()
    # 步骤1在任何torch/模型导入之前固定可见设备；步骤2按显式阶段进入本轮隔离模块。
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu if a.gpu is not None and a.stage in ('select','qa') else ''
    if a.stage=='engineering':
        from videoqa_access.acceptance import engineering
        result=engineering(root)
    else:
        from videoqa_access import runtime
        result=getattr(runtime,{'coordinate':'coordinate','select':'select_worker','inputs':'freeze_inputs',
            'qa':'qa_worker','replay':'replay','report':'report','audit':'audit'}[a.stage])(root,a)
    if result is not None:print(result,flush=True)


if __name__=='__main__':main()
