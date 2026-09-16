"""验证离线增益归因的数值闭合与共同前缀，不调用模型。"""
import copy
import numpy as np
from videoqa_methods.algorithm import compete,reselect
from videoqa_audit.stage_attribution import coarse_audit,refinement_audit,divergence,parts,margin


def fixture(scores,features,edges,lam,count=2):
    """以原选择器构造合成日志，再由独立审计器重放。"""
    c=[dict(candidate_index=i,source_pts=i*4+1,timestamp_seconds=i+.25) for i in range(len(scores))]
    segments=[dict(segment_id=j,start=a,end=b,start_pts=0 if a==0 else c[a]['source_pts'],representative_score=max(scores[a:b])) for j,(a,b) in enumerate(zip(edges,edges[1:]))]
    selected,quotas,trace=compete(c,scores,features,segments,lam,{'redundancy_coefficient':.2},count=count)
    return {'pool':dict(candidates=c,scores=scores,segments=segments,initial_count=len(c),lambda_value=lam,selected_indices=selected,coarse_selected=selected,quotas=quotas,competition_trace=trace,video={'time_base':'1/4','start_pts':0},hotspot_trace=[])}


def test_coarse_replay_and_zero_quota():
    """没有配额的片段与本段局部排名失败分开报告。"""
    f=np.eye(6,dtype=np.float32);r=fixture([.9,.8,.7,.1,0.,0.],f,[0,3,6],.2)
    audit=coarse_audit(r,f,{1,17},.2)
    assert audit['maximum_gain_error']==0
    assert audit['targets']['1']['coarse_selected']
    assert audit['targets']['17']['stage']=='segment_received_zero_quota'


def test_same_segment_divergence_has_no_coverage_difference():
    """同段的A/B首分歧由相对相关性/去重权衡产生，覆盖项差严格为0。"""
    f=np.array([[1,0],[1,0],[0,1]],dtype=np.float32)
    a=fixture([.9,.85,.55],f,[0,3],.5);b=fixture([.9,.85,.55],f,[0,3],.2)
    d=divergence(a,b,f,.2)
    assert d['step']==2 and d['same_segment']
    assert d['under_A']['gap']<0<d['under_B']['gap']
    assert d['under_A']['winner_minus_target']['coverage']==0


def test_different_segment_divergence():
    """不同段时原覆盖压力确实可以改变分支。"""
    f=np.eye(3,dtype=np.float32)
    a=fixture([.9,.8,.6],f,[0,2,3],.5);b=fixture([.9,.8,.6],f,[0,2,3],.8)
    d=divergence(a,b,f,.2)
    assert not d['same_segment'] and d['under_A']['gap']<0<d['under_B']['gap']


def test_refinement_uses_empty_local_prefix():
    """重选的第一帧重复度为0，不能错误沿用粗选前缀。"""
    f=np.array([[1,0],[1,0],[0,1]],dtype=np.float32)
    r=fixture([.9,.8],f[:2],[0,2],.5,count=1);p=r['pool']
    p['candidates'].append(dict(candidate_index=2,source_pts=2,timestamp_seconds=.5));p['scores'].append(.95)
    final,trace=reselect(p['candidates'],p['scores'],f,p['segments'],p['quotas'],p['coarse_selected'],2,.5,{'redundancy_coefficient':.2})
    p['selected_indices']=final;p['reselection_trace']=trace
    result=refinement_audit(r,f,{1,2},.2)
    assert result['maximum_gain_error']==0 and not result['targets']['1']['final_selected']
    assert result['targets']['2']['events'][0]['margin']['winner']['redundancy_term']==0
