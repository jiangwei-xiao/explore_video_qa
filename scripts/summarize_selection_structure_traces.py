"""同粗池全量离线轨迹汇总：只描述受限空间，不把约束差异当成事实损失。"""
from collections import Counter
import argparse
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def describe_control(parent,control,trace,parent_trace):
    """接口：同粗池源帧与事后映射→约束统计；不接受题目答案或语义事实。"""
    def frames(item):
        return [(f['source_frame_index'],f['source_pts']) for f in item['selection']['selected_frames']]
    pkeys,ckeys=frames(parent),frames(control);pset,cset=set(pkeys),set(ckeys);k=parent['actual_frame_count']
    if k!=len(pset) or k!=len(cset) or control['actual_frame_count']!=k:
        raise ValueError('Same-coarse actual budgets differ')
    mapping=trace['posthoc_parent_mapping'];regions=mapping['regions'];protected=parent_trace['protected_indices']
    if sum(r['parent_quota'] for r in regions)!=k or sum(r['control_count'] for r in regions)!=k:
        raise ValueError('Mapped regional budgets are not closed')
    max_quota=k-len(regions)+1;out_of_prefix=0;out_of_feasible_prefix=0;zero=0;prefix_compatible=True
    # 步骤1对照受限前缀，不假定被移除的源帧就是唯一必要证据。
    local_prefix={str(r['region_id']):r['prefix'] for r in parent_trace['coarse_prefix_trace']['regions']}
    for r in regions:
        q=r['control_count'];prefix=local_prefix[str(r['region_id'])]
        zero+=q==0
        if q<1 or q>max_quota or set(r['selected_indices'])!=set(prefix[:q]):prefix_compatible=False
        for pos in r['selected_prefix_positions'].values():
            out_of_prefix+=pos is None
            out_of_feasible_prefix+=(pos is None or pos>max_quota)
    transferred=sum(max(0,r['control_count']-r['parent_quota']) for r in regions)
    source_equal=pkeys==ckeys and parent['selection'].get('video')==control['selection'].get('video')
    return dict(actual_frames=k,source_equal=source_equal,parent_overlap=len(pset&cset),
        parent_overlap_fraction=len(pset&cset)/k,region_count=len(regions),
        parent_anchor_count=len(protected),retained_anchors=len(mapping['anchor_retained']),
        retained_topk_seeds=len(mapping['seed_retained']),
        zero_parent_regions=zero,transferred_region_slots=transferred,
        outside_recorded_prefix=out_of_prefix,outside_feasible_parent_prefix=out_of_feasible_prefix,
        within_parent_feasible_family=prefix_compatible,
        same_region_counts_different_input=(transferred==0 and not source_equal),
        wfs_branch=trace.get('fallback'),
        limitation='Source/constraint proxies only; not semantic evidence loss or isolated module causality')


def aggregate(records):
    """接口：按实际分母汇总结构统计，不读取任何方法正确率。"""
    if not records:return dict(n=0)
    numeric=('parent_overlap_fraction','region_count','parent_anchor_count','retained_anchors',
             'retained_topk_seeds','transferred_region_slots','outside_recorded_prefix','outside_feasible_parent_prefix')
    return dict(n=len(records),same_source=sum(r['source_equal'] for r in records),
        within_parent_feasible_family=sum(r['within_parent_feasible_family'] for r in records),
        any_anchor_lost=sum(r['retained_anchors']<r['parent_anchor_count'] for r in records),
        any_parent_region_zero=sum(r['zero_parent_regions']>0 for r in records),
        same_region_counts_different_input=sum(r['same_region_counts_different_input'] for r in records),
        means={k:sum(r[k] for r in records)/len(records) for k in numeric},
        actual_frames=dict(Counter(str(r['actual_frames']) for r in records)),
        wfs_branches=dict(Counter(r['wfs_branch'] for r in records if r['wfs_branch'] is not None)))


def summarize(root):
    """接口：已通过全量离线验收的32400份选择/轨迹→结构报告；新增选择与模型调用均0。"""
    from videoqa_full.state import rows
    from videoqa_audit import selection_structure as method
    from videoqa_audit import selection_structure_run as study
    root=Path(root);summary=study.read(root/'offline_summary.json')
    if (summary['status']!='passed' or summary['parent_replays']!=5400 or summary['control_selections']!=10800
            or summary['protocol_sha256']!=study.sha(root/'offline_protocol.json')):
        raise ValueError('Complete offline acceptance is required')
    output=root/'posthoc_structure/summary.json'
    if output.exists():
        value=study.read(output)
        if value['offline_summary_sha256']!=study.sha(root/'offline_summary.json') or value['auditor_sha256']!=study.sha(__file__):
            raise ValueError('Completed posthoc structure provenance differs')
        if study.sha(root/'posthoc_structure/per_question.json')!=value['per_question_sha256']:
            raise ValueError('Completed per-question structure record changed')
        for relative,h in value['source_exports'].items():
            if study.sha(root/relative)!=h:raise ValueError('Completed structure source changed')
        return value
    data=rows();records=[];sources={}
    def checked(relative):
        path=root/relative;h=study.sha(path)
        if h!=summary['exports'][relative]:raise ValueError('Frozen offline artifact changed: '+relative)
        sources[relative]=h;return study.read(path)
    # 步骤2只扫描已实际引用的小型JSON，不解码视频、不读取特征、不重新执行选择器。
    for profile in study.PROFILES:
        for row in data:
            q=row['question_id'];parent=checked(f'selections/{profile}/{method.PARENT}/{q}.json')
            p_trace=checked(f'traces/{profile}/{method.PARENT}/{q}.json')
            for control in method.CONTROLS:
                item=checked(f'selections/{profile}/{control}/{q}.json')
                trace=checked(f'traces/{profile}/{control}/{q}.json')
                detail=describe_control(parent,item,trace,p_trace)
                records.append(dict(question_id=q,video_id=row['video_id'],stratum=row['stratum'],
                                    profile=profile,control=control,**detail))
    grouped={}
    for profile in study.PROFILES:
        grouped[profile]={}
        for control in method.CONTROLS:
            subset=[r for r in records if r['profile']==profile and r['control']==control]
            grouped[profile][control]={'all2700':aggregate(subset)}
            grouped[profile][control].update({s:aggregate([r for r in subset if r['stratum']==s]) for s in ('short','medium','long')})
    study.freeze(root/'posthoc_structure/per_question.json',records)
    value=dict(status='passed',profiles=grouped,question_control_records=len(records),
        source_exports=sources,offline_summary_sha256=study.sha(root/'offline_summary.json'),
        per_question_sha256=study.sha(root/'posthoc_structure/per_question.json'),
        auditor_sha256=study.sha(__file__),new_selections=0,new_model_calls=0,
        limitation='No semantic loss rate or causal attribution; no QA outcomes used')
    study.freeze(output,value)
    return value


def main():
    """接口：仅提供运行目录的只读汇总，不开放选择参数或题目筛选。"""
    import json
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);args=parser.parse_args()
    value=summarize(Path(args.run_dir));print(json.dumps({k:v for k,v in value.items() if k!='source_exports'},ensure_ascii=False))


if __name__=='__main__':main()
