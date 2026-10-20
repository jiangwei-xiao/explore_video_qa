"""复用首批结束后的CPU容量：仅完成原固定总体中剩余的源资格，不改变研究机制。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from preindex_longvideo_trial_assets import publish_result
from videoqa_set_value.longvideo_assets import inventory
from videoqa_set_value.longvideo_trial import index_fingerprint
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc


def pending_sources(root):
    """接口：只读已完成时长初筛和八包凭证；逆序取原队列剩余项，不读题目或答案。"""
    root=Path(root);eligible={p.stem:read_json(p) for p in (root/'duration_checks').glob('*.json')}
    eligible={k:v for k,v in eligible.items() if v['eligible']};ordered=[];sources={}
    # 步骤1：输入总体与原协调器相同，不能增加分片、修改时长或跳过已知身份隔离。
    for entry in inventory()[3:]:
        path=ROOT/'data/videos/longvideo_reason/extraction'/(entry['name']+'.json')
        value=read_json(path)
        if value['archive_sha256']!=entry['sha256']:raise ValueError('Fixed archive receipt changed')
        sources[str(path)]=sha256(path)
        for member in value['members']:
            vid=member['video_id']
            if vid in eligible:
                record=eligible[vid]
                if any(record[k]!=member[k] for k in ('video_id','path','sha256','bytes')):
                    raise ValueError('Duration/source identity mismatch')
                ordered.append(record)
    if len(ordered)!=len(eligible) or len({x['video_id'] for x in ordered})!=len(ordered):
        raise ValueError('Original qualified-duration population differs')
    # 步骤2：主队列从前推进，这里从另一端计算；完成集合按源ID判断，不能累加重复计数。
    pending=[x for x in reversed(ordered) if not (root/'identity_checks'/(x['video_id']+'.json')).exists()]
    return pending,sources


def check_one(job):
    """接口：逐项重新检查主缓存；已完成则复用，否则调用同一冻结全解码/指纹函数。"""
    root,stage,record=Path(job[0]),Path(job[1]),job[2]
    # 步骤1：队列等待期间主流程可能已经完成该源，避免再解码。
    if (root/'identity_checks'/(record['video_id']+'.json')).exists():
        return index_fingerprint((str(root),record)),None
    # 步骤2：独立暂存保证同源竞争时不会共写PTS数组；由原子发布器核对并发身份。
    result=index_fingerprint((str(stage),record))
    return result,str(stage/'identity_checks'/(record['video_id']+'.json'))


def run(root):
    """接口：等首批CPU任务退出后保持4+4进程上限；不启动BLIP、教师或训练。"""
    root=Path(root);stage=root/'remaining_source_checks'
    with exclusive(stage/'worker.lock'):
        # 步骤1：先导/暂停/停止均由原协调器控制，本工具不能越过这些边界。
        while not (root/'preliminary_checks/preindex_completed.json').exists():
            if any((root/name).exists() for name in ('PAUSE.json','STOP.json','manifest.json')):
                result=dict(status='cancelled_before_start',model_calls=0,utc=utc())
                durable(stage/'completed.json',result);return result
            durable(stage/'progress.json',dict(stage='waiting_for_first_cpu_batch',pid=os.getpid(),sid=os.getsid(0),model_calls=0,utc=utc()))
            time.sleep(5)
        previous=read_json(root/'preindex_identity.json')['pid']
        while True:
            try:command=Path('/proc',str(previous),'cmdline').read_bytes()
            except FileNotFoundError:break
            if b'preindex_longvideo_trial_assets.py' not in command or str(root.resolve()).encode() not in command:break
            time.sleep(1)
        if any((root/name).exists() for name in ('PAUSE.json','STOP.json','manifest.json')):
            result=dict(status='cancelled_before_start',model_calls=0,utc=utc())
            durable(stage/'completed.json',result);return result
        pending,sources=pending_sources(root);began=time.monotonic()
        durable(stage/'plan.json',dict(source_ids=[x['video_id'] for x in pending],sources=sources,
            workers=4,ordering='reverse_original_remaining_source_order',labels_read=False,
            code_sha256=sha256(__file__),original_index_code_sha256=sha256(ROOT/'src/videoqa_set_value/longvideo_trial.py'),utc=utc()))
        # 步骤2：所有结果是原总体的缓存；不修改主游标、manifest或研究阶段门。
        completed=0;published=0;reused=0
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(check_one,(str(root),str(stage),r)) for r in pending]
            for future in as_completed(futures):
                result,path=future.result();completed+=1
                if path is None:reused+=1
                else:published+=publish_result(root,path,result)
                durable(stage/'progress.json',dict(stage='remaining_source_checks',completed=completed,total=len(pending),
                    published=published,reused=reused,seconds=time.monotonic()-began,
                    pid=os.getpid(),sid=os.getsid(0),model_calls=0,utc=utc()))
        # 步骤3：这里只结束CPU辅助工作，不声明训练总体或学习结果成立。
        result=dict(status='completed_remaining_source_checks',sources=len(pending),published=published,reused=reused,
            seconds=time.monotonic()-began,model_calls=0,utc=utc())
        durable(stage/'completed.json',result);return result


def main():
    """接口：显式研究运行目录；输出保持隔离并可追溯。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True)
    print(run(parser.parse_args().run_root))


if __name__=='__main__':main()
