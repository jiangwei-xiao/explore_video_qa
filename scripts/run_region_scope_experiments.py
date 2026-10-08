"""exp27独立阶段入口：区域职责解耦，冻结离线产物后执行双配置开发验证。"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_audit.region_scope_experiments import offline
from videoqa_full.state import durable,utc


def main():
    """接口：显式阶段与恢复目录；没有独立M、窗口参数或额外版本搜索开关。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=('offline','prepare-inputs','qa','report','audit','all'),required=True)
    parser.add_argument('--profile',choices=('qwen32','llava8'))
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():parser.error('Invalid run id')
    root=ROOT/'outputs/methods'/args.run_id
    try:
        # 步骤1：离线阶段只消费旧映射和特征；GPU阶段必须经过独立review门槛。
        if args.stage=='offline':offline(root,args.workers)
        else:
            from videoqa_audit.region_scope_runtime import dispatch
            dispatch(root,args.stage,args.profile,args.gpus)
    except BaseException as error:
        # 步骤2：保留阶段异常，禁止增加调用预算或静默修改输入条件。
        durable(root/('failure_'+args.stage+'.json'),dict(error=repr(error),utc=utc()));raise


if __name__=='__main__':main()
