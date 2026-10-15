"""exp33约束回归：测试状态保护和邻接合法性，不调用任何问答/评分模型。"""
import copy
from fractions import Fraction
import numpy as np
import pytest
from videoqa_methods.adjacent_context_repair import select_adjacent_repair,propose_adjacent_repair


def data():
    """接口：粗帧前缀与细帧后缀交错在真实时间轴，确保不按数组编号判断邻接。"""
    points=[25,225,425,625,75,275,475]
    frames=[dict(source_pts=p,source_frame_index=p,candidate_index=i,requested_fraction=str(Fraction(p,100)),
                 requested_seconds=p/100,timestamp_seconds=p/100) for i,p in enumerate(points)]
    pool=dict(candidates=frames,initial_count=4,initial_scores=[.9,.8,.7,.6],
              plan=dict(seed_indices=[0,3],regions=[dict(anchor_index=0)]))
    f=np.array([[1,0],[0,1],[-1,0],[0,-1],[1,0],[0,1],[-1,0]],dtype=np.float32)
    return pool,f,[0,1,2,3]


def test_one_replacement_preserves_other_slots_and_input():
    p,f,s=data();old=copy.deepcopy(p)
    out,t=select_adjacent_repair(p,f,s,4)
    assert out==[0,4,2,3] and t['changed_slots']==[1] and p==old and s==[0,1,2,3]
    assert set(t['protected_indices'])<=set(out)


def test_non_seed_coarse_frame_can_request():
    p,f,s=data();p['plan']['seed_indices']=[3];p['plan']['regions']=[dict(anchor_index=3)]
    out,t=select_adjacent_repair(p,f,s,4)
    assert t['winner']['anchor_index']==0 and out==[0,4,2,3]


def test_seed_and_region_anchor_not_removed():
    p,f,s=data();p['plan']['seed_indices']=[1,2,3]
    out,t=select_adjacent_repair(p,f,s,4)
    assert out==s and not t['changed']
    assert 'following_slot_is_protected' in t['reason_counts']


def test_selected_direct_predecessor_or_successor_suppresses_request():
    p,f,_=data()
    out,t=select_adjacent_repair(p,f,[0,4,1,3],4)
    assert not any(x['anchor_index'] in (0,1) for x in t['proposals'])
    assert t['reason_counts']['direct_time_neighbour_already_selected']>=2


def test_removed_fine_frame_has_no_automatic_immunity():
    p,f,_=data();p['candidates'].append(dict(source_pts=150,source_frame_index=150,candidate_index=7,
        requested_fraction='3/2',timestamp_seconds=1.5));f=np.vstack((f,[0,1])).astype(np.float32)
    out,t=select_adjacent_repair(p,f,[0,7,2,3],4)
    assert t['winner']['replaced_index']==7 and out==[0,4,2,3]


def test_only_one_swap_even_when_many_proposals_exist():
    p,f,s=data();out,t=select_adjacent_repair(p,f,s,4)
    assert len(t['proposals'])==2 and len([i for i,(a,b) in enumerate(zip(s,out)) if a!=b])==1


def test_itm_priority_and_equal_score_continuity():
    p,f,s=data();p['initial_scores'][1]=.95
    out,t=select_adjacent_repair(p,f,s,4)
    assert t['winner']['anchor_index']==1 and out==[0,1,5,3]
    p['initial_scores'][1]=.9
    assert select_adjacent_repair(p,f,s,4)[1]['winner']['anchor_index']==0


@pytest.mark.parametrize('change',['wrong_grid','missing_target','zero_delta'])
def test_target_mapping_and_positive_delta_required(change):
    p,f,s=data()
    if change=='wrong_grid':p['candidates'][4]['requested_fraction']='3/2'
    elif change=='missing_target':
        p['candidates'][4].pop('requested_fraction');p['candidates'][4].pop('requested_seconds')
    else:f[4]=[0,1]
    t=propose_adjacent_repair(p,f,s,4)
    assert not any(x['anchor_index']==0 for x in t['proposals'])


def test_actual_source_time_not_average_fps_and_no_source_duplicates():
    p,f,s=data();p['candidates'][4]['source_pts']=26;p['candidates'][4]['source_frame_index']=90000
    assert select_adjacent_repair(p,f,s,4)[0]==[0,4,2,3]
    p['candidates'][4]['source_pts']=25
    with pytest.raises(ValueError,match='Duplicate source'):select_adjacent_repair(p,f,s,4)


def test_short_full_pool_no_duplicate_padding():
    p,f,_=data();s=sorted(range(len(p['candidates'])),key=lambda i:p['candidates'][i]['source_pts'])
    out,t=select_adjacent_repair(p,f,s,32)
    assert out==s and len(out)==7 and not t['changed']


def test_interface_has_no_answer_or_question_argument():
    import inspect
    assert list(inspect.signature(select_adjacent_repair).parameters)==['pool','features','selected_indices','K']


def test_small_roundoff_is_not_a_continuity_signal():
    p,f,s=data();f[4]=[1e-7,1]
    t=propose_adjacent_repair(p,f,s,4)
    assert not any(x['anchor_index']==0 for x in t['proposals'])


def test_stopped_offline_branch_exposes_no_generation_stage():
    import subprocess,sys
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/run_adjacent_context_repair.py'
    result=subprocess.run([sys.executable,str(script),'--run-id','unit_test_missing_exp33_gate','--stage','qa'],capture_output=True,text=True)
    assert result.returncode!=0 and 'invalid choice' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr and 'Loading checkpoint' not in result.stderr
