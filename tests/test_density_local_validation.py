"""局部验证器的纯逻辑验收，不调用模型或生成新答案。"""
import inspect
import numpy as np
import pytest
from videoqa_runtime.common import ROOT,read_json
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import build_regions,select_joint
from videoqa_audit.density_local_validation import supported_selection

CFG=read_json(ROOT/'configs/retrieval_density_v1.json')


def pool(n):
    """构造有精确源身份的规则候选，用于算法而非画面语义验证。"""
    rows=[dict(candidate_index=i,source_pts=4*i+1,source_frame_index=4*i+1,
               timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(n)]
    return rows,dict(time_base='1/4',start_pts=0,duration_fraction=str(n),duration_seconds=float(n))


def test_supported_candidate_beats_unsupported_outlier():
    rows,video=pool(3);cfg=dict(CFG,seed_budget=2,frame_budget=2)
    plan=build_regions(rows,[.9,.8,.1],video,cfg)
    features=np.array([[1,0,0],[.8,.6,0],[0,0,1]],np.float32)
    assert select_joint(rows,features,plan,video,cfg)[0]==[0,2]
    chosen,trace=supported_selection(rows,features,plan,video,cfg)
    assert chosen==[0,1] and trace['supports']['0']['2']==0


def test_support_references_include_non_anchor_seeds():
    rows,video=pool(3);cfg=dict(CFG,seed_budget=2,frame_budget=2)
    plan=build_regions(rows,[.9,.8,.1],video,cfg)
    features=np.array([[1,0,0],[0,1,0],[0,0,1]],np.float32)
    chosen,trace=supported_selection(rows,features,plan,video,cfg)
    assert chosen==[0,1] and trace['supports']['0']['1']==1


def test_new_old_labels_do_not_change_selection():
    rows,video=pool(6);plan=build_regions(rows,[.9,.8,.7,.6,.5,.4],video,CFG)
    features=np.eye(6,dtype=np.float32)
    reference=supported_selection(rows,features,plan,video,CFG)
    changed=[dict(r,origin='new' if i%2 else 'old') for i,r in enumerate(rows)]
    assert supported_selection(changed,features,plan,video,CFG)==reference


def test_fixed_original_quotas_remain_fixed():
    rows,video=pool(40);cfg=dict(CFG,seed_budget=3,frame_budget=5)
    scores=[.9 if i in (0,1,30) else .1 for i in range(40)]
    plan=build_regions(rows,scores,video,cfg)
    features=np.tile(np.array([1.,0.],np.float32),(40,1));features[1]=[0,1];features[31]=[0,1]
    chosen,trace=supported_selection(rows,features,plan,video,cfg,{'0':2,'1':3})
    assert len(chosen)==5 and trace['region_budgets']=={'0':2,'1':3}
    assert set(trace['protected_indices'])<=set(chosen)


@pytest.mark.parametrize('quotas',[{'0':0},{'0':10},{'1':3}])
def test_bad_fixed_quota_rejected(quotas):
    rows,video=pool(3);plan=build_regions(rows,[.9,.8,.7],video,CFG)
    with pytest.raises(ProtocolError):supported_selection(rows,np.eye(3,dtype=np.float32),plan,video,CFG,quotas)


def test_zero_utility_fills_unique_budget_by_source_order():
    rows,video=pool(6);cfg=dict(CFG,frame_budget=4)
    plan=build_regions(rows,[.9,.8,.7,.6,.5,.4],video,cfg)
    chosen,trace=supported_selection(rows,np.tile(np.array([1.,0.],np.float32),(6,1)),plan,video,cfg)
    assert chosen==[0,1,2,3] and all(r['winner']['priority']==0 for r in trace['rounds'])


def test_selector_has_no_answers_or_manual_facts_input():
    assert set(inspect.signature(supported_selection).parameters)=={'candidates','features','plan','video','cfg','quotas'}
