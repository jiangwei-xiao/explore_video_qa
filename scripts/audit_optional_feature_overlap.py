"""exp23重合细帧的独立RGB与特征核验，读取旧PNG，不调用模型。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE

SOURCE=ROOT/'outputs/analysis/optional_seed64_pool200_20261004_r1'


def inspect_question(job):
    """接口：按源PTS和源帧号对应旧/新细帧，不能直接比较候选编号。"""
    from PIL import Image
    run,q=job;old=read_json(BASE/'pools'/(q+'.json'));new=read_json(SOURCE/'pools'/(q+'.json'))
    old_features=np.load(BASE/'features'/(q+'.npy'),allow_pickle=False,mmap_mode='r')
    new_features=np.load(run/'features'/(q+'.npy'),allow_pickle=False,mmap_mode='r')
    lookup={(c['source_pts'],c['source_frame_index']):c['candidate_index'] for c in new['candidates']}
    recorded={}
    # 步骤1：新批次实际RGB哈希必须唯一并对应冻结源身份。
    for path in (run/'calls/fine'/q).glob('*.json'):
        record=read_json(path)
        if record['status']!='completed':raise ValueError('未完成视觉批次')
        for frame,digest in zip(record['frames'],record['rgb_sha256']):
            key=(frame['source_pts'],frame['source_frame_index'])
            if key in recorded:raise ValueError('新细帧重复计算')
            recorded[key]=digest
    errors=[];maximum=0.;checked=0
    # 步骤2：旧PNG由原局部连续解码保存，与本轮精确读取路径独立比较。
    for asset in old['new_assets']:
        i=asset['candidate_index'];frame=old['candidates'][i];key=(frame['source_pts'],frame['source_frame_index'])
        if key not in lookup or key not in recorded:raise ValueError('旧细帧在新池中缺失')
        path=Path(asset['path'])
        if not path.is_absolute():path=ROOT/path
        if sha256(path)!=asset['sha256']:raise ValueError('旧PNG身份改变')
        with Image.open(path) as image:
            digest=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
        error=float(np.max(np.abs(old_features[i]-new_features[lookup[key]])))
        maximum=max(maximum,error);checked+=1
        if digest!=recorded[key] or error>1e-6:
            errors.append(dict(source_frame=frame,rgb_equal=digest==recorded[key],feature_error=error))
    return dict(question_id=q,checked=checked,maximum_feature_error=maximum,mismatches=errors)


def main(run,wait):
    """接口：只等待同一已知准备进程；完整准备后核验重合10733张细帧。"""
    # 步骤1：没有完成文件时验证实际PID，不因日志超时重新启动任何任务。
    while not (run/'summary.json').exists():
        status=read_json(run/'status.json')
        if not wait or status['status']=='paused':raise RuntimeError('准备尚未完成')
        proc=Path('/proc')/str(status['pid'])/'cmdline'
        if not proc.exists() or 'prepare_optional_seed64_features.py' not in proc.read_bytes().decode():
            raise RuntimeError('准备进程不存在')
        time.sleep(2)
    summary=read_json(run/'summary.json')
    if summary['status']!='passed':raise ValueError('准备未验收')
    ids=[r['question_id'] for r in read_json(SOURCE/'summary.json')['records']]
    # 步骤2：不重新编码图像，不把容差或丢失PNG问题静默忽略。
    with ThreadPoolExecutor(max_workers=4) as executor:
        records=list(executor.map(inspect_question,[(run,q) for q in ids]))
    result=dict(status='passed' if not any(r['mismatches'] for r in records) else 'failed',
        source_summary_sha256=sha256(run/'summary.json'),code_sha256=sha256(__file__),records=records,
        checked=sum(r['checked'] for r in records),new_model_calls=0,
        maximum_feature_error=max(r['maximum_feature_error'] for r in records),
        scope='旧新重合细帧；不把此核验冒充新增9185源帧的逐帧语义审核')
    if result['checked']!=10733:raise ValueError('重合细帧分母变化')
    path=run/'overlap_audit.json'
    if path.exists() and read_json(path)!=result:raise ValueError('审核恢复变化')
    if not path.exists():durable(path,result)
    print(result['status'],result['checked'],result['maximum_feature_error'],flush=True)
    if result['status']!='passed':raise ValueError('RGB或特征不一致，停止后续放行')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();main(args.run,args.wait)
