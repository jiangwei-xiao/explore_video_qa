"""固定近重复诊断的时间与防链式漂移约束。"""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('repeat_check',Path(__file__).resolve().parents[1]/'scripts/check_temporal_repeat_competition.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_pairwise_constraint_prevents_chain_drift():
    """相邻各相似但首尾不相似时，不能链式并为一组。"""
    theta=np.asarray([0.,.2,.4]);features=np.stack([np.cos(theta),np.sin(theta)],axis=1)
    frames=[dict(source_pts=i,candidate_index=i) for i in range(3)]
    assert module.group_runs([2,0,1],features,frames,dict(start_pts=0,time_base='1/2'))==[[0,1],[2]]


def test_time_gap_and_endpoint():
    """一秒端点允许合并，大于一秒即使像素相同也分开。"""
    frames=[dict(source_pts=p,candidate_index=i) for i,p in enumerate([0,2,5])]
    features=np.asarray([[1.,0.]]*3)
    assert module.group_runs([0,1,2],features,frames,dict(start_pts=0,time_base='1/2'))==[[0,1],[2]]
