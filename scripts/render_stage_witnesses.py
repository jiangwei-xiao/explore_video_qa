#!/usr/bin/env python3
"""人工指定候选见证的CPU核查：仅评价与溯源，不进入自动选帧。"""
import argparse
import sys
from pathlib import Path
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable
from videoqa_lmms_bridge.frames import FrameProvider
from videoqa_lmms_baseline.methods import SOURCE


def render(run,q,indices):
    """接口：输入人工核查候选编号，保存原帧、排名与窗口资格；不能反馈给选择器。"""
    p=SOURCE/'llava/pools'/(q+'.json');pool=read_json(p);dest=run/'witnesses'/q;dest.mkdir(parents=True,exist_ok=True)
    # 步骤1：按原有分数和稳定同分规则复算排名，不重新BLIP评分。
    c=pool['candidates'];scores=pool['initial_scores'];order=sorted(range(pool['initial_count']),key=lambda i:(-scores[i],c[i]['source_pts'],i))
    output=[]
    for i in indices:
        f=c[i];images,hashes=FrameProvider(dict(video=pool['video'],selected_frames=[f])).decode()
        try:images[0].save(dest/f'{i}_source.png')
        finally:
            for image in images:image.close()
        # 步骤2：时间落入窗口、粗候选资格和最终保留分别记录，不把临近等同同一事实。
        t=Fraction(f['source_pts']-pool['video']['start_pts'])*Fraction(pool['video']['time_base'])
        regions=[g['region_id'] for g in pool['plan']['regions'] if Fraction(g['left_fraction'])<=t<=Fraction(g['right_fraction'])]
        output.append(dict(candidate_index=i,frame=f,rank=order.index(i)+1 if i in order else None,score=scores[i] if i<len(scores) else None,
            region_ids=regions,eligible=i in pool['selection_trace']['eligible_indices'],selected=i in pool['selected_indices'],rgb_sha256=hashes[0]))
    durable(dest/'records.json',dict(pool_sha256=sha256(p),purpose='human-directed evidence audit only',new_model_calls=0,records=output));print(q,output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--qid',required=True);p.add_argument('--indices',required=True);a=p.parse_args()
    assert Path(a.run_id).name==a.run_id and Path(a.qid).name==a.qid
    render(ROOT/'outputs/analysis'/a.run_id,a.qid,[int(i) for i in a.indices.split(',')])
