"""局部数值耦合诊断只验证原评分，不实现新算法。"""
import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('coupling',Path(__file__).resolve().parents[1]/'scripts/analyze_anchor_score_coupling.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_same_anchor_prefers_two_thirds():
    """同一锚点时中等相似度胜过更近和更远帧，这是原式属性而非实现故障。"""
    s=[1.,.4,2/3,.9]
    f=np.asarray([[x,math.sqrt(1-x*x)] for x in s],dtype=np.float32)
    c=[dict(source_pts=i) for i in range(4)]
    region=dict(members=list(range(4)),local_steps=[dict(candidate_index=2,local_score=2/(3*math.sqrt(3)))])
    result=module.first_step(f,region,dict(anchor_index=0,seed_indices=[0]),[1.,.2,.2,.2],c)
    assert result['winner']==2 and result['coupled_candidate_count']==3


def test_frozen_winner_mismatch_rejected():
    f=np.eye(2,dtype=np.float32)
    region=dict(members=[0,1],local_steps=[dict(candidate_index=0,local_score=0.)])
    with pytest.raises(ValueError,match='winner'):
        module.first_step(f,region,dict(anchor_index=0,seed_indices=[0]),[1.,0.],[dict(source_pts=0),dict(source_pts=1)])
