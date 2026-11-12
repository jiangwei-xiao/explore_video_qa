"""独立统计审计边界：视频内抵消不得被当成独立题重抽。"""
import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def auditor():
    path=Path(__file__).resolve().parents[1]/'scripts/audit_selection_structure_completion.py'
    spec=importlib.util.spec_from_file_location('independent_structure_audit_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_clustered_opposite_changes_cancel_in_every_resample(auditor):
    data=[dict(question_id='001-1',video_id='001',stratum='short'),
          dict(question_id='001-2',video_id='001',stratum='short')]
    value=auditor.independent_pair(data,{'001-1':True,'001-2':False},{'001-1':False,'001-2':True})
    assert value['stratified_video_cluster_bootstrap_95']==[0.,0.]
    assert value['cells']['left_only']==['001-1'] and value['cells']['right_only']==['001-2']


def test_unanimous_improvement_is_one_hundred_percentage_points(auditor):
    data=[dict(question_id=str(i),video_id=str(i),stratum=s) for i,s in enumerate(('short','medium','long'))]
    value=auditor.independent_pair(data,{str(i):True for i in range(3)},{str(i):False for i in range(3)})
    assert value['difference_pp']==100.
    assert value['stratified_video_cluster_bootstrap_95']==[100.,100.]


def test_wilson_known_binomial_interval(auditor):
    assert auditor.independent_wilson(10,20)==pytest.approx([.2992980081982123,.7007019918017877],abs=1e-12)
