"""exp28独立后台入口：单规则双开发验证，通过固定门槛后Qwen32全量。"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,argparse,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,utc,exclusive
from videoqa_audit.conditional_merge_experiments import offline


def stage_root(run_id):
    """接口：全量后缀仅标记运行阶段，不改变方法ID，旧运行目录不覆盖。"""
    return ROOT/('outputs/full_videoqa' if run_id.endswith('__full') else 'outputs/methods')/run_id


def initialize(root):
    """接口：阶段清单与离线来源事前固定；恢复不重新选模型或开发题。"""
    full=root.name.endswith('__full');dev=ROOT/'outputs/methods'/root.name.removesuffix('__full')
    meta=dict(population='full2700' if full else 'development200',profiles=['qwen32'] if full else ['qwen32','llava8'],
              offline_root=str(dev),development_root=str(dev),full_target_profile='qwen32',clear_drop_correct_questions=4)
    root.mkdir(parents=True,exist_ok=True);path=root/'study_stage.json'
    if path.exists():
        if read_json(path)!=meta:raise ValueError('Study stage changed')
    else:durable(path,meta)


def input_gate(root):
    """接口：完整离线身份核验后放行输入准备；事实指标不按得分挑选候选版本。"""
    from videoqa_audit.region_scope_review import witness_report
    source=Path(read_json(root/'study_stage.json')['offline_root']);summary=read_json(source/'offline_summary.json')
    if summary['status']!='passed' or summary['records']!=5400:raise ValueError('Offline completion missing')
    for p,h in summary['files'].items():
        if sha256(p)!=h:raise ValueError('Offline artifact changed')
    if source==root:witness_report(root)
    value=dict(status='passed',kind='engineering_input_gate_not_semantic_sufficiency',
               offline_protocol_sha256=sha256(source/'offline_protocol.json'),offline_summary_sha256=sha256(source/'offline_summary.json'),utc=utc())
    if source!=root:
        dev=read_json(source/'summary.json')
        if not dev['full_gate_passed']:raise ValueError('Development accuracy stop gate')
        value['development_summary_sha256']=sha256(source/'summary.json')
    path=root/'review_gate.json'
    if path.exists():
        saved=read_json(path)
        if any(saved[k]!=v for k,v in value.items() if k!='utc'):raise ValueError('Input gate changed')
    else:durable(path,value)


def main():
    """接口：固定阶段与GPU，不提供参数扫描；已启动异常只记录并暂停。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',required=True,choices=('offline','prepare-inputs','qa','report','audit','all','night'))
    parser.add_argument('--profile',choices=('qwen32','llava8'))
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():parser.error('Invalid run ID')
    root=stage_root(args.run_id);initialize(root)
    try:
        if args.stage=='offline':offline(root,args.workers)
        elif args.stage=='night':
            # 步骤1：单一版本离线与双配置开发；后续全量条件在正式生成前冻结。
            with exclusive(root/'night.lock'):
                offline(root,args.workers);input_gate(root)
                from videoqa_audit.conditional_merge_runtime import dispatch
                dispatch(root,'all',None,args.gpus)
                development=read_json(root/'summary.json')
                if not development['full_gate_passed']:
                    durable(root/'night_status.json',dict(status='candidate_stopped_development_drop',development=str(root/'summary.json'),utc=utc()))
                    return
                # 步骤2：固定Qwen32全量，导入本轮开发答案，绝不按新分数切换配置。
                full=stage_root(args.run_id+'__full');initialize(full);input_gate(full)
                durable(root/'night_status.json',dict(status='full_running',full_root=str(full),utc=utc()))
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run-id',full.name,'--stage','all','--gpus',*args.gpus],cwd=ROOT,check=True)
                durable(root/'night_status.json',dict(status='experiment_completed',full_root=str(full),utc=utc()))
        else:
            from videoqa_audit.conditional_merge_runtime import dispatch
            dispatch(root,args.stage,args.profile,args.gpus)
    except BaseException as error:
        durable(root/('failure_'+args.stage+'.json'),dict(error=repr(error),utc=utc()));raise


if __name__=='__main__':main()
