"""冻结exp33四单元开发测试入口；旧离线保护失败保留，本阶段允许已知风险小测。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_audit.adjacent_context_development import preflight,prepare_all,prepare_unit,execute_all,run_unit,worker,audit_all,report


def main():
    """接口：prepare在CPU完成，qa/worker明确物理卡；all自动完成后台阶段切换。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--stage',choices=('all','prepare','prepare-unit','qa','run-unit','worker','audit','report'),default='all')
    p.add_argument('--dataset',choices=('videomme','mlvu'));p.add_argument('--profile',choices=('llava8','qwen32'))
    p.add_argument('--gpu');p.add_argument('--gpus')
    args=p.parse_args()
    if not args.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    root=ROOT/'outputs/methods'/args.run_id
    # 步骤1：完整环境启动协调器，单元子进程自动选择已冻结的后端环境。
    if args.stage in ('all','prepare'):preflight(root);prepare_all(root)
    if args.stage in ('all','qa'):execute_all(root)
    elif args.stage=='prepare':return
    elif args.stage=='audit':audit_all(root)
    elif args.stage=='report':report(root)
    else:
        if not args.dataset or not args.profile:p.error('Cell requires dataset and profile')
        if args.stage=='prepare-unit':prepare_unit(root,args.dataset,args.profile)
        elif args.stage=='run-unit':
            gpus=(args.gpus or '').split(',')
            if not 1<=len(gpus)<=2 or len(set(gpus))!=len(gpus) or any(g not in map(str,range(8)) for g in gpus):p.error('One or two unique physical GPUs required')
            run_unit(root,args.dataset,args.profile,gpus)
        else:
            if args.gpu not in list(map(str,range(8))) or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu:p.error('Worker must see exactly its physical GPU')
            worker(root,args.dataset,args.profile,args.gpu)


if __name__=='__main__':main()
