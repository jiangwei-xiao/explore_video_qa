"""触发核心补查范围测试：真实源边界、去重与冻结特征映射，无模型调用。"""
from fractions import Fraction
import numpy as np
import pytest
from videoqa_methods.budget_contract import budget_config,select_exp07
from videoqa_methods.density_selection import build_regions,refinement_targets
from videoqa_methods.region_refinement_scope import refinement_windows,restrict_refinement_pool


def make_pool(seed_times=(2.25,3.25,4.25,10.25),fine=(7.75,10.75),sparse=False):
    """测试接口：完整1FPS候选或合法稀疏VFR源映射，创建原合并窗口记录。"""
    times=[0.25+i for i in range(19)]
    if sparse:times=[t for t in times if not 5.25<t<10.25]
    rows=[dict(candidate_index=i,source_pts=round(t*4),source_frame_index=round(t*4),timestamp_seconds=t,
               requested_seconds=6.25 if sparse and t==10.25 else t) for i,t in enumerate(times)]
    scores=[0.05]*len(rows)
    for rank,t in enumerate(seed_times):scores[times.index(t)]=0.99-rank*0.01
    video=dict(start_pts=0,time_base='1/4',duration_fraction='19')
    cfg=budget_config(len(seed_times));plan=build_regions(rows,scores,video,cfg);n=len(rows)
    for t in fine:
        rows.append(dict(candidate_index=len(rows),source_pts=round(t*4),source_frame_index=round(t*4),
                         timestamp_seconds=t,requested_seconds=t,requested_fraction=str(Fraction(t))))
    reports=[]
    for region in plan['regions']:
        maps=[]
        for target in refinement_targets(region,video,cfg) if region['refine'] else []:
            possible=[r for r in rows if Fraction(r['timestamp_seconds'])>=target]
            if not possible:
                maps.append(dict(target_fraction=str(target),status='no_source_in_window'))
                continue
            frame=min(possible,key=lambda r:r['source_pts'])
            maps.append(dict(target_fraction=str(target),status='added' if frame['candidate_index']>=n else 'duplicate_source',
                             candidate_index=frame['candidate_index'],source_pts=frame['source_pts'],source_frame_index=frame['source_frame_index']))
        reports.append(dict(region_id=region['region_id'],mappings=maps))
    pool=dict(video=video,candidates=rows,initial_count=n,initial_scores=scores,plan=plan,refinement=reports)
    return pool,np.eye(len(rows),dtype=np.float32)


def test_mixed_core_does_not_inherit_refinement_and_features_are_exact():
    p,f=make_pool();new,nf,t=restrict_refinement_pool(p,f,4)
    assert p['plan']['regions'][0]['core_seed_counts']==[3,1]
    assert new['plan']==p['plan'] and new['candidates'][:p['initial_count']]==p['candidates'][:p['initial_count']]
    assert [x['timestamp_seconds'] for x in new['candidates'][p['initial_count']:]]==[7.75]
    assert np.array_equal(nf,f[new['parent_candidate_indices']])
    assert t['removed_parent_indices']==[p['initial_count']+1]
    assert len(select_exp07(new,nf,4)[0])==4


def test_touching_trigger_windows_merge_and_clip_at_start():
    p,f=make_pool(seed_times=(0.25,1.25,2.25,10.25,11.25,12.25),fine=())
    ws=refinement_windows(p['plan'],p['video'],6)
    assert len(ws)==1 and ws[0]['left_fraction']=='0' and ws[0]['right_fraction']=='65/4'
    assert ws[0]['core_ids']==[0,1]


def test_untriggered_pool_is_unchanged_and_selection_equivalent():
    p,f=make_pool(seed_times=(2.25,3.25,10.25,11.25),fine=())
    new,nf,_=restrict_refinement_pool(p,f,4)
    assert new['candidates']==p['candidates'] and np.array_equal(nf,f)
    assert select_exp07(new,nf,4)[0]==select_exp07(p,f,4)[0]


def test_vfr_source_crossing_scope_is_rejected_even_for_existing_coarse_frame():
    p,f=make_pool(seed_times=(0.25,1.25,2.25,10.25),fine=(1.75,10.75),sparse=True)
    new,_,_=restrict_refinement_pool(p,f,4)
    item=next(m for r in new['refinement'] for m in r['mappings'] if m['target_fraction']=='23/4')
    assert item['status']=='outside_new_scope'
    assert any(x['timestamp_seconds']==10.25 for x in new['candidates'][:new['initial_count']])


def test_missing_mapping_stops_instead_of_reconstructing_source():
    p,f=make_pool();p['refinement'][0]['mappings']=[]
    with pytest.raises(ValueError,match='Missing frozen target'):restrict_refinement_pool(p,f,4)


def test_reused_mapping_keeps_parent_and_global_identity():
    p,f=make_pool();p['original_candidate_indices']=[i+100 for i in range(len(f))]
    for region in p['refinement']:
        for m in region['mappings']:
            i=m['candidate_index'];m.update(status='reused' if i>=p['initial_count'] else 'duplicate_source',original_index=i+100)
            del m['candidate_index'];del m['source_pts'];del m['source_frame_index']
    new,nf,_=restrict_refinement_pool(p,f,4)
    assert new['original_candidate_indices']==[i+100 for i in new['parent_candidate_indices']]
    assert np.array_equal(nf,f[new['parent_candidate_indices']])


def test_video_end_clipping_preserves_last_legal_fine_frame():
    p,f=make_pool(seed_times=(2.25,16.25,17.25,18.25),fine=(18.75,))
    windows=refinement_windows(p['plan'],p['video'],4)
    assert windows[0]['right_fraction']=='19'
    new,_,_=restrict_refinement_pool(p,f,4)
    assert new['candidates'][-1]['timestamp_seconds']==18.75
    assert all(Fraction(t)<=19 for r in new['refinement'] for t in r['targets'])


def test_short_pool_has_distinct_seed_and_actual_frame_counts_without_padding():
    p,f=make_pool(seed_times=tuple(0.25+i for i in range(19)),fine=(0.75,))
    new,nf,_=restrict_refinement_pool(p,f,32)
    selected,trace=select_exp07(new,nf,32)
    assert len(selected)==len(set(selected))==20
    assert trace['budget_contract']==dict(frame_budget=32,seed_count=19,actual_frame_count=20)


def test_disjoint_trigger_windows_remain_separate_and_keep_exact_features():
    p,f=make_pool(seed_times=(0.25,1.25,2.25,15.25,16.25,17.25),fine=(1.75,16.75))
    windows=refinement_windows(p['plan'],p['video'],6)
    assert len(windows)==2 and Fraction(windows[0]['right_fraction'])<Fraction(windows[1]['left_fraction'])
    new,nf,_=restrict_refinement_pool(p,f,6)
    assert [c['timestamp_seconds'] for c in new['candidates'][new['initial_count']:]]==[1.75,16.75]
    assert np.array_equal(nf,f[new['parent_candidate_indices']])
