"""网络恢复薄入口：4MiB断点子段与原64MiB接收器兼容，实验与算法保持冻结。"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：先可执行纯字节验收；正式运行仅接触固定公开数据和下载暂存目录。"""
    from videoqa_set_value import longvideo_transport as t
    from videoqa_set_value import longvideo_assets as a
    p=argparse.ArgumentParser();p.add_argument('--acceptance-only',action='store_true')
    p.add_argument('--state-dir',default='/tmp/longvideo-relay-20261019-r1')
    p.add_argument('--asset-root',default='/home/admin/projects/explore_video_qa/data/videos/longvideo_reason')
    p.add_argument('--socket',default='/tmp/videoqa-mlvu-night-20261015-reconnect.sock')
    p.add_argument('--remote',default='admin@10.1.64.8');p.add_argument('--port',type=int,default=10087)
    args=p.parse_args();remote=t.Remote(args.asset_root,args.socket,args.remote,args.port)
    if args.acceptance_only:
        entry=a.inventory()[3];state=remote.query('asset-state',entry)
        expected=state['chunks']['0']['sha256'];blocks=t.download_batch([(entry,0)],args.state_dir)
        block=blocks[0][2]
        if block['sha256']!=expected:raise ValueError('Segmented range differs from existing verified source bytes')
        value=dict(status='acceptance_passed',bytes=block['bytes'],sha256=block['sha256'],uploaded=False,
            segment_bytes=t.SEGMENT_BYTES,model_calls=0)
        a.atomic_json(Path(args.state_dir)/'acceptance.json',value)
    else:value=t.run(remote,args.state_dir)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
