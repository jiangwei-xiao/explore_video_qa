"""原生批次闭包与末批边界：不能通过放宽FP32容差掩盖批次形状变化。"""
import importlib.util
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('native_probe',ROOT/'scripts/run_blip_joint_native_probe.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_native_tail_and_fine_are_separate():
    pool=dict(initial_count=60,candidates=[dict(source_pts=i) for i in range(64)])
    groups=module.native_groups(pool,[0,58,60,63])
    assert [g['ids'] for g in groups]==[list(range(16)),list(range(48,60)),[60,63]]


def test_duplicate_or_outside_bank_rejected():
    pool=dict(initial_count=2,candidates=[dict(source_pts=i) for i in range(3)])
    for bank in ([],[0,0],[3],[-1]):
        with pytest.raises(ValueError):module.native_groups(pool,bank)
