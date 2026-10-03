"""窗外提议只依赖时间和原问题分数，不读取题型或答案。"""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('outside_support',Path(__file__).resolve().parents[1]/'scripts/probe_external_support.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def pool(scores):
    """接口：一秒一源帧，原窗口闭区间[2,4]。"""
    return dict(candidates=[dict(candidate_index=i,source_pts=i,source_frame_index=i) for i in range(len(scores))],
        initial_count=len(scores),initial_scores=scores,video=dict(start_pts=0,time_base='1'),
        plan=dict(regions=[dict(left_fraction='2',right_fraction='4')]))


def test_closed_window_and_inclusive_suppression():
    p=pool([1.]*12);chosen,_=module.propose(p)
    assert chosen==[0,5,10] and not set(chosen)&{2,3,4}


def test_zero_scores_are_not_new_evidence():
    assert module.propose(pool([0.]*8))[0]==[]
