"""全量M64/K32结构预检；只构建种子与窗口，不准备细帧或调用模型。"""
import argparse
from fractions import Fraction
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,utc
from videoqa_audit.query_input_probe import BASE
from videoqa_methods.density_selection import frame_key,source_time


def geometry(candidates,scores,video,seed_count):
    """接口：复算原2秒核心连接及±4秒扩窗；保留超过32区域的事实，不静默合并。"""
    if len(candidates)!=len(scores) or not candidates or seed_count<1:raise ValueError('输入规模不符')
    # 步骤1：完全沿用原稳定种子排序和目标网格连接，只省略与几何无关的全池成员扫描。
    seeds=sorted(range(len(scores)),key=lambda i:(-scores[i],*frame_key(candidates[i])))[:seed_count]
    cores=[]
    for i in sorted(seeds,key=lambda i:frame_key(candidates[i])):
        if not cores or Fraction(str(candidates[i]['requested_seconds']))-Fraction(str(candidates[cores[-1][-1]]['requested_seconds']))>2:
            cores.append([])
        cores[-1].append(i)
    # 步骤2：扩窗按真实PTS，重叠或共享端点按原规则合并，不引入预算。
    regions=[];end=Fraction(video['duration_fraction'])
    for k,ids in enumerate(cores):
        left=max(Fraction(0),source_time(candidates[ids[0]],video)-4)
        right=min(end,source_time(candidates[ids[-1]],video)+4)
        if regions and left<=Fraction(regions[-1]['right_fraction']):
            regions[-1]['right_fraction']=str(max(right,Fraction(regions[-1]['right_fraction'])))
            regions[-1]['core_ids'].append(k)
        else:regions.append(dict(left_fraction=str(left),right_fraction=str(right),core_ids=[k]))
    for r in regions:
        r['seed_indices']=[i for k in r['core_ids'] for i in cores[k]]
        r['refine']=max(len(cores[k]) for k in r['core_ids'])>=3
    return dict(seed_indices=seeds,regions=regions)


def signature(plan):
    """提取冻结实现可逐项核对的几何字段，不忽略核心合并信息。"""
    return dict(seed_indices=plan['seed_indices'],regions=[{k:r[k] for k in
        ['left_fraction','right_fraction','core_ids','seed_indices','refine']} for r in plan['regions']])


def main(run):
    """接口：先对全2700题复现M32，再报告M64预算可行性；不生成新问答。"""
    import os
    manifest=ROOT/'data/manifests/video_mme_full_2700.json'
    dev=ROOT/'data/manifests/video_mme_development_200_20260920.json'
    rows=read_json(manifest)['rows'];dev_ids={r['question_id'] for r in read_json(dev)['rows']}
    if len(rows)!=2700 or len(dev_ids)!=200:raise ValueError('冻结名单分母变化')
    sources={str(p):sha256(p) for p in [Path(__file__),manifest,dev,ROOT/'src/videoqa_methods/density_selection.py']}
    records=[]
    # 步骤1：所有初始候选与原评分复用，旧细帧不作为种子。
    for n,row in enumerate(rows,1):
        q=row['question_id'];path=BASE/'pools'/(q+'.json');sources[str(path)]=sha256(path)
        pool=read_json(path);c=pool['candidates'][:pool['initial_count']];s=pool['initial_scores'];v=pool['video']
        control=geometry(c,s,v,32)
        if control!=signature(pool['plan']):raise ValueError('M32几何未逐项复现: '+q)
        candidate=geometry(c,s,v,64)
        if not set(control['seed_indices'])<=set(candidate['seed_indices']):raise ValueError('种子嵌套关系不符')
        def covered(plan):return sum(Fraction(r['right_fraction'])-Fraction(r['left_fraction']) for r in plan['regions'])
        records.append(dict(question_id=q,stratum=row['stratum'],development=q in dev_ids,
            old_regions=len(control['regions']),new_regions=len(candidate['regions']),
            old_window_seconds=float(covered(control)),new_window_seconds=float(covered(candidate)),
            protected_anchor_capacity_ok=len(candidate['regions'])<=32,plan=candidate))
        if n%100==0:durable(run/'status.json',dict(status='running',completed=n,total=2700,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
    # 步骤2：如实列出不满足K32的题，不删题、回退或改变窗口间隔。
    summary=dict(questions=2700,control_replayed=2700,new_model_calls=0,
        over_budget_ids=[r['question_id'] for r in records if not r['protected_anchor_capacity_ok']],
        development_over_budget_ids=[r['question_id'] for r in records if r['development'] and not r['protected_anchor_capacity_ok']],
        old_regions_total=sum(r['old_regions'] for r in records),new_regions_total=sum(r['new_regions'] for r in records),
        maximum_new_regions=max(r['new_regions'] for r in records),
        limitation='纯种子/窗口结构，不是完整M64候选池，不报告新选帧或准确率')
    result=dict(sources=sources,records=records,summary=summary)
    path=run/'analysis.json'
    if path.exists():
        if read_json(path)!=result:raise ValueError('恢复来源或分析改变')
    else:durable(path,result)
    durable(run/'status.json',dict(status='completed',completed=2700,total=2700,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();args.run.mkdir(parents=True,exist_ok=True);main(args.run)
