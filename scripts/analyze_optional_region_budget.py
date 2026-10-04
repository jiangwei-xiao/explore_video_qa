"""冻结exp12前缀及覆盖曲线上允许零区域配额的离线对照，不修改生产DP。"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget

CURRENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def optional_allocation(curves,weights,budget):
    """接口：精确求解n_r≥0、总和为K；同分优先较早区域更多名额，与旧DP一致。

    C_r(0)=0；非零区域仍使用原以锚点开头的局部前缀，不从中间任意取帧。
    """
    if not curves or len(curves)!=len(weights) or budget<0:raise ValueError('范围不符')
    # 步骤1：只将原枚举下界1改为0，不加新惩罚项或阈值。
    states={0:(0.,())}
    for curve,weight in zip(curves,weights):
        if not curve or curve[0]!=0 or weight<=0:raise ValueError('覆盖原点或权重错误')
        updated={}
        for used,(value,allocation) in states.items():
            for n in range(0,min(len(curve)-1,budget-used)+1):
                proposal=(value+weight*curve[n],allocation+(n,))
                if used+n not in updated or proposal>updated[used+n]:updated[used+n]=proposal
        states=updated
    # 步骤2：预算必须填满，零配额不是减少总帧数的捷径。
    if budget not in states:raise ValueError('容量不足')
    value,allocation=states[budget]
    return list(allocation),value


def main(run):
    """接口：全2700题重放原DP，再比较唯一的零下界改动；问答及模型均0调用。"""
    manifest=ROOT/'data/manifests/video_mme_full_2700.json'
    devpath=ROOT/'data/manifests/video_mme_development_200_20260920.json'
    rows=read_json(manifest)['rows'];dev={r['question_id'] for r in read_json(devpath)['rows']}
    sources={str(p):sha256(p) for p in [manifest,devpath,Path(__file__),ROOT/'src/videoqa_methods/prefix_coverage_budget.py']}
    records=[]
    # 步骤1：所有曲线与密度权重均取自冻结运行，不重新抽样或评分。
    for row in rows:
        q=row['question_id'];p=CURRENT/'traces'/(q+'.json');sources[str(p)]=sha256(p);trace=read_json(p)
        regions=trace['regions'];curves=[r['coverage_curve'] for r in regions];weights=[r['density_weight'] for r in regions]
        old=[trace['region_budgets'][str(r['region_id'])] for r in regions];budget=sum(old)
        restored,old_value=allocate_prefix_budget(curves,weights,budget)
        if restored!=old:raise ValueError('原DP配额未复现: '+q)
        new,value=optional_allocation(curves,weights,budget)
        if value+1e-12<old_value:raise ValueError('放宽可行域后目标降低')
        records.append(dict(question_id=q,development=q in dev,stratum=row['stratum'],old=old,new=new,
            objective_gain=value-old_value,zero_regions=sum(n==0 for n in new),
            old_selection=sorted(i for r,n in zip(regions,old) for i in r['prefix'][:n]),
            new_selection=sorted(i for r,n in zip(regions,new) for i in r['prefix'][:n])))
    # 步骤2：只报告结构及代理目标，不附用历史预测、不当作新方法精度。
    changed=[r for r in records if r['old']!=r['new']]
    summary=dict(questions=len(records),changed=len(changed),development_changed=sum(r['development'] for r in changed),
        changed_ids=[r['question_id'] for r in changed],zero_region_total=sum(r['zero_regions'] for r in records),new_model_calls=0)
    output=dict(sources=sources,records=records,summary=summary,
        limitation='仅冻结M32的预算可行域对照；不证明M64细帧池或最终选择已实现')
    path=run/'analysis.json'
    if path.exists():
        if read_json(path)!=output:raise ValueError('恢复来源或结果变化')
    else:run.mkdir(parents=True,exist_ok=True);durable(path,output)
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
