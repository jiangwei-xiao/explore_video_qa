"""有界帧组探针必须沿用真实PTS映射。"""
import importlib.util
from pathlib import Path
from fractions import Fraction
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('framegroup_prepare',Path(__file__).resolve().parents[1]/'scripts/prepare_framegroup_probe.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_variable_pts_first_at_or_after_target():
    rows=module.map_targets(np.array([10,11,13,16]),dict(start_pts=10,time_base='1/10'),[Fraction(1,20),Fraction(1,5),Fraction(1,2)])
    assert [r['source_frame_index'] for r in rows]==[1,2,3]


def test_duplicate_mapping_pauses_instead_of_padding():
    with pytest.raises(ValueError):module.map_targets(np.array([0,10]),dict(start_pts=0,time_base='1'),[Fraction(1),Fraction(2)])
