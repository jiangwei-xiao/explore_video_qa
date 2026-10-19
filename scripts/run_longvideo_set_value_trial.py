"""独立长视频集合价值试点：固定资产中转、资格冻结与分阶段一次性执行。"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：资产阶段不导入模型依赖；训练/验证必须经过独立门槛与调用预算。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['relay','asset-state','asset-receive','asset-finalize','asset-relay-status',
        'prepare','run','worker','freeze','audit'])
    parser.add_argument('--run-id',default='set_value_longvideo__llava8__train800_val400__20261019_r1')
    parser.add_argument('--asset-root',default='/home/admin/projects/explore_video_qa/data/videos/longvideo_reason')
    parser.add_argument('--file');parser.add_argument('--offset',type=int);parser.add_argument('--length',type=int)
    parser.add_argument('--sha256');parser.add_argument('--gpu');parser.add_argument('--phase')
    parser.add_argument('--socket',default='/tmp/videoqa-mlvu-night-20261015-reconnect.sock')
    parser.add_argument('--remote',default='admin@10.1.64.8');parser.add_argument('--port',type=int,default=10087)
    parser.add_argument('--relay-state',default='/tmp/longvideo-relay-20261019-r1')
    parser.add_argument('--metadata-only',action='store_true')
    parser.add_argument('--relay-status',choices=['running','failed','completed'])
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum():parser.error('Invalid isolated run ID')
    from videoqa_set_value import longvideo_assets as assets
    if args.stage=='relay':
        result=assets.relay(args.asset_root,args.remote,args.socket,args.port,args.relay_state,metadata_only=args.metadata_only)
    elif args.stage=='asset-state':result=assets.asset_state(args.asset_root,args.file)
    elif args.stage=='asset-receive':
        result=assets.receive_chunk(args.asset_root,args.file,args.offset,args.length,args.sha256,sys.stdin.buffer)
    elif args.stage=='asset-finalize':result=assets.finalize_asset(args.asset_root,args.file)
    elif args.stage=='asset-relay-status':
        import time
        if args.relay_status is None:parser.error('Missing relay status')
        result=dict(status=args.relay_status,unix=time.time())
        assets.atomic_json(Path(args.asset_root)/'relay_status.json',result)
    else:
        from videoqa_set_value import longvideo_trial as trial
        root=ROOT/'outputs/set_value'/args.run_id
        if args.stage=='freeze':
            from videoqa_set_value.longvideo_runtime import freeze_inputs
            if args.phase not in ('pilot','train_rest','val'):parser.error('Invalid freeze phase')
            result=freeze_inputs(root,args.phase)
        elif args.stage=='worker':
            if args.gpu not in ('0','1','2','3') or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu:
                parser.error('Only the assigned idle physical GPU0–3 may be exposed')
            result=trial.worker(root,args.asset_root,args.phase,args.gpu)
        elif args.stage=='prepare':result=trial.prepare(root,args.asset_root)
        elif args.stage=='audit':result=trial.audit(root)
        else:result=trial.run(root,args.asset_root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
