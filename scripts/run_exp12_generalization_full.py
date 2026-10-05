"""冻结exp12三项全量验证；后台协调无需Codex/SSH连接存活。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from pathlib import Path
import argparse
import subprocess
import sys
import re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_full.state import durable, exclusive, utc
from videoqa_generalization.inputs import prepare_inputs
from videoqa_generalization.run import prepare_stage, execute, audit, QPY, LPY, STAGES


def synthetic_llava(run, allowed):
    """接口：正式LLaVA路径的1/11/16帧合成验收，独立记账且不生成真实题答案。"""
    from videoqa_runtime.baseline_run import available_gpus
    from videoqa_runtime.common import offline_environment
    import time
    while True:
        try:free,_=available_gpus(allowed)
        except RuntimeError:free=[]
        if free: break
        durable(run/'status.json',dict(status='waiting_resources',utc=utc())); time.sleep(15)
    os.environ['CUDA_VISIBLE_DEVICES']=free[0]; offline_environment()
    import torch
    from PIL import Image
    from videoqa_runtime.llava_backend import LlavaBackend
    from videoqa_lmms_bridge.llava_adapter import FrozenInputLlavaVid
    from videoqa_lmms_bridge.bridge import synthetic_row
    from videoqa_generalization.llava_observe import observe
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    backend=LlavaBackend(); adapter=FrozenInputLlavaVid(backend); results=[]
    # 步骤1：正式入口、视频预算与观察器统一；横竖输入不改变384预处理协议。
    for n,size in [(1,(320,240)),(11,(720,1280)),(16,(896,672))]:
        path=run/'raw'/f'{n}.json'
        if path.exists(): raise ValueError('Synthetic prior attempt requires audit; no automatic replay')
        images=[Image.new('RGB',size,(50,i*7,150)) for i in range(n)]
        s=dict(video=dict(path='synthetic.mp4',duration_seconds=float(n)),
               selected_frames=[dict(timestamp_seconds=float(i)) for i in range(n)])
        state={}; result=observe(backend,adapter,images,s,synthetic_row(n),state,path,dict(synthetic_frames=n))
        for image in images:image.close()
        results.append(dict(n=n,size=list(size),state=state,result=result))
    # 步骤2：合成只作运行验收，不按输出内容挑配置。
    durable(run/'synthetic_acceptance.json',dict(status='passed',checks=results,synthetic_generations=3,real_generations=0,utc=utc()))


def main():
    """接口：显式运行编号/GPU与恢复，异常暂停全流水线，正式上限7900次。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['all','prepare','synthetic_qwen','synthetic_llava',*STAGES,'report'],default='all')
    parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id):raise ValueError('Invalid run ID')
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<=set(map(str,range(8))):raise ValueError('Invalid GPU scope')
    root=ROOT/'outputs/full_videoqa'/args.run_id; root.mkdir(parents=True,exist_ok=True)
    if any(root.glob('*/calls')) and not args.resume:raise ValueError('Existing calls require explicit resume')
    if args.stage=='all':
        with exclusive(root/'pipeline.lock'):
            durable(root/'launcher_identity.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            # 步骤1：按已批准顺序，各模型/环境使用独立子进程，结束后释放GPU。
            for stage in ['prepare','synthetic_qwen','synthetic_llava','qwen16','llava16','wfs32','report']:
                py=LPY if stage in ['synthetic_llava','llava16'] else QPY
                durable(root/'pipeline_status.json',dict(status='running',stage=stage,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                command=[py,str(Path(__file__).resolve()),'--run-id',args.run_id,'--stage',stage,'--gpus',*args.gpus]
                if args.resume:command.append('--resume')
                try:subprocess.run(command,cwd=ROOT,check=True)
                except BaseException as exc:
                    durable(root/'pipeline_status.json',dict(status='paused',stage=stage,error=repr(exc),utc=utc()));raise
            durable(root/'pipeline_status.json',dict(status='completed',new_answer_calls=7900,utc=utc()))
    elif args.stage=='prepare':prepare_inputs(root)
    elif args.stage=='synthetic_qwen':
        from videoqa_generalization.run import verify_wfs_inputs
        verify_wfs_inputs(root)
        from videoqa_budget32 import run as qrun
        qrun.GPUS=tuple(args.gpus)
        run=root/'synthetic/qwen'
        if not (run/'synthetic_acceptance.json').exists():qrun.synthetic(run)
    elif args.stage=='synthetic_llava':
        run=root/'synthetic/llava'
        if not (run/'synthetic_acceptance.json').exists():synthetic_llava(run,args.gpus)
    elif args.stage in STAGES:
        # 步骤2：两种合成均通过后才可冻结正式协议；完成恢复不加载模型。
        from videoqa_runtime.common import read_json
        for kind in ['qwen','llava']:
            if read_json(root/f'synthetic/{kind}/synthetic_acceptance.json')['status']!='passed':raise ValueError('Synthetic gate incomplete')
        prepare_stage(root,args.stage,args.gpus); execute(root/args.stage); audit(root/args.stage)
    else:
        from videoqa_generalization.report import summarize
        summarize(root)


if __name__=='__main__':main()
