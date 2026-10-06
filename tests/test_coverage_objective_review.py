"""纯结构证据路径：编号迁移、预算可行性和区域多对多对应。"""
import copy
import pytest
from videoqa_audit.coverage_objective_review import compare_allocations,trace_witness_path,region_correspondence


def fixture():
    video=dict(path='v',start_pts=0,time_base='1/10')
    pool=dict(video=video,initial_count=6,candidates=[dict(source_pts=i*10,source_frame_index=i) for i in range(6)],
        plan=dict(seed_indices=[0,3]))
    regions=[dict(region_id=0,members=[0,1,2],reference_coarse_indices=[0,1,2],prefix=[0,2,1],density_weight=1),
        dict(region_id=1,members=[3,4,5],reference_coarse_indices=[3,4,5],prefix=[3,5,4],density_weight=1)]
    trace=dict(regions=regions,region_budgets={'0':1,'1':3})
    return pool,trace


def test_allocation_is_single_factor_and_symmetric_transfer():
    pool,a=fixture();b=copy.deepcopy(a);b['region_budgets']={'0':2,'1':2}
    r=compare_allocations(pool,a,b)
    assert r['transferred_slots']==1 and r['selected_overlap']==3 and r['budget']==4
    b['regions'][0]['prefix']=[0,1,2]
    with pytest.raises(ValueError,match='Upstream'):compare_allocations(pool,a,b)


def test_source_identity_mapping_ignores_old_candidate_number():
    pool,t=fixture();r=trace_witness_path(pool,t,[(2,20),(2,20)])
    assert len(r['paths'])==1 and r['paths'][0]['candidate_index']==2
    assert r['paths'][0]['state']=='quota_reachable' and r['paths'][0]['additional_slots']==1


def test_missing_pool_is_not_missing_prefix():
    pool,t=fixture();r=trace_witness_path(pool,t,[(9,90)])
    assert r['paths'][0]['state']=='outside_legal_pool' and not r['exact_hit']
    t['regions'][0]['prefix']=[0,2]
    assert trace_witness_path(pool,t,[(1,10)])['paths'][0]['state']=='outside_prefix'


def test_positive_anchor_floor_limits_quota():
    pool,t=fixture();t['region_budgets']={'0':1,'1':1}
    r=trace_witness_path(pool,t,[(2,20)])['paths'][0]
    assert r['state']=='beyond_feasible_quota' and r['max_feasible_quota']==1


def test_alternative_witness_selected_prevents_fact_loss():
    pool,t=fixture();r=trace_witness_path(pool,t,[(0,0),(2,20)])
    assert r['exact_hit'] and r['any_quota_reachable']


def test_cross_budget_split_not_region_id_match():
    pool,a=fixture();b=dict(regions=[dict(region_id=7,reference_coarse_indices=list(range(6)))])
    edges=region_correspondence(pool,b,pool,a)
    assert len(edges)==2 and all(e['relation']=='split' and e['left_region']==7 for e in edges)


def test_fixed_panel_keeps_empty_slots_and_unique_videos():
    """不足类别不拿其他题补位；外部视频优先级不能破坏固定两诊断。"""
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/analyze_coverage_objectives.py'
    spec=importlib.util.spec_from_file_location('coverage_entry',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=[dict(question_id=q,video_id=q.split('-')[0]) for q in ['409-3','556-1','001-1','002-1']]
    records={p:{r['question_id']:dict(correctness_cell='both_wrong',same_input=True,
        answers={'BASE-Uniform':dict(correct=False),'REF-WFS-SB':dict(correct=False)}) for r in data} for p in ['qwen32','llava8']}
    for q in ['001-1','002-1']:records['qwen32'][q]['correctness_cell']='left_only'
    panel=module.freeze_panel(data,records,{'001'})
    assert panel['questions'][2]['question_id']=='002-1'
    assert len(panel['questions'])==3 and sum(s['question_id'] is None for s in panel['slots'])==9
