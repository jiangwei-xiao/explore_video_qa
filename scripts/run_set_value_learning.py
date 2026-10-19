"""独立数据轻量集合价值学习：可恢复分阶段入口，不启动历史评测问答。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：准备、冻结、教师、训练、验证、报告分阶段，worker必须暴露唯一分配卡。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--stage',required=True,choices=['preflight','run','wait_data_and_run','selection_worker','teacher_worker','freeze_inputs','fit','predict','report','audit'])
    p.add_argument('--gpu');p.add_argument('--data-pid',type=int);args=p.parse_args()
    if not args.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    from videoqa_set_value import runtime as r
    root=ROOT/'outputs/set_value'/args.run_id
    if args.stage.endswith('_worker'):
        if args.gpu not in map(str,range(8)) or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu:p.error('Worker must expose assigned GPU only')
        if args.stage=='selection_worker':r.selection_worker(root,args.gpu)
        else:r.teacher_worker(root,args.gpu)
    elif args.stage=='preflight':r.preflight(root)
    elif args.stage=='run':r.run(root)
    elif args.stage=='wait_data_and_run':
        if args.data_pid is None:p.error('Data transition requires a verified process ID')
        r.wait_data_and_run(root,args.data_pid)
    elif args.stage=='freeze_inputs':r.freeze_inputs(root)
    elif args.stage=='fit':r.fit(root)
    elif args.stage=='predict':r.predict_validation(root)
    elif args.stage=='audit':r.verify(root,full=True);r.audit_phase(root,'train');r.audit_phase(root,'val')
    else:
        from videoqa_set_value.report import report
        report(root)


if __name__=='__main__':main()
