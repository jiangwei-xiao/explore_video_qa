"""下载期间预计算独立原片的PTS/指纹；不在部分总体上选训练或验证视频。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def publish_result(root,staged,result):
    """接口：原子发布完整独立视频资格缓存，不覆盖原协调器先完成的结果。"""
    from videoqa_runtime.common import read_json
    target=Path(root)/'identity_checks'/(result['video_id']+'.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(staged,target)
    except FileExistsError:
        old=read_json(target)
        # 步骤1：路径可因独立计算目录不同而异，但源身份、资格和视觉指纹必须相同。
        keys=('video_id','sha256','bytes','eligible','reason','fingerprint')
        if any(old.get(k)!=result.get(k) for k in keys):raise ValueError('Concurrent qualification differs')
        return False
    return True


def run(root):
    """接口：复用冻结全解码/八帧指纹函数；完整总体划分仍只由原协调器执行。"""
    from videoqa_set_value.longvideo_trial import index_fingerprint
    from videoqa_runtime.common import read_json,sha256
    from videoqa_full.state import durable,exclusive,utc
    root=Path(root);stage=root/'preliminary_checks';start=time.monotonic()
    with exclusive(stage/'preindex.lock'):
        report=read_json(root/'preliminary_container_report.json')
        for path,expected in report['receipts'].items():
            if sha256(path)!=expected:raise ValueError('Preliminary archive receipt changed')
        records=[read_json(p) for p in sorted((stage/'duration_checks').glob('*.json'))]
        records=[r for r in records if r['eligible']]
        # 步骤2：所有时长合格原片都计算，不依据哈希排名或任何答案提前挑选1200视频。
        completed=0;published=0;eligible=0
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(index_fingerprint,(str(stage),r)):r for r in records}
            for future in as_completed(futures):
                result=future.result();vid=result['video_id']
                published+=publish_result(root,stage/'identity_checks'/(vid+'.json'),result)
                eligible+=bool(result['eligible']);completed+=1
                elapsed=time.monotonic()-start
                durable(stage/'preindex_progress.json',dict(completed=completed,total=len(records),
                    published=published,eligible=eligible,seconds=elapsed,pid=os.getpid(),sid=os.getsid(0),
                    eta_seconds=elapsed*(len(records)-completed)/completed,model_calls=0,utc=utc()))
        # 步骤3：本地原片验收不等于全总体近重复隔离，禁止在此发布manifest或启动教师。
        value=dict(status='completed_partial_source_checks',videos=completed,eligible=eligible,
            full_population_qualified=False,split_frozen=False,model_calls=0,seconds=time.monotonic()-start,utc=utc())
        durable(stage/'preindex_completed.json',value)
        return value


def main():
    """接口：只有本轮已有运行目录；所有模型和训练代码保持原样。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True)
    args=parser.parse_args();print(run(args.run_root))


if __name__=='__main__':main()
