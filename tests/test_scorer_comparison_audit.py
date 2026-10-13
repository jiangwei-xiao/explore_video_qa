"""独立评分审计回归：有理数时钟、VFR首次映射与实际特征维度。"""
import numpy as np
import pytest
from videoqa_full.state import durable
from videoqa_audit.scorer_comparison_audit import verify_source_mapping,actual_feature_dim


def test_vfr_uses_first_source_after_rational_target():
    """接口：VFR不能按平均FPS估计；目标0.5秒对应实际0.9秒的首张帧。"""
    v=dict(time_base='1/10',start_pts=0)
    f=dict(source_frame_index=2,source_pts=9,requested_fraction='1/2',requested_seconds=.5,timestamp_seconds=.9)
    verify_source_mapping(v,[f],np.array([0,3,9,11]))


def test_later_source_is_not_accepted_as_first():
    """接口：时间合法但不是第一张的帧，仍必须被映射验收拒绝。"""
    v=dict(time_base='1/10',start_pts=0)
    f=dict(source_frame_index=3,source_pts=11,requested_seconds=.5,timestamp_seconds=1.1)
    with pytest.raises(ValueError,match='first source'):
        verify_source_mapping(v,[f],np.array([0,3,9,11]))


def test_nonzero_origin_and_fractional_tick():
    """接口：非零起点和非整数目标tick用有理数上取整，不能发生浮点边界偏差。"""
    v=dict(time_base='1/30',start_pts=100)
    f=dict(source_frame_index=1,source_pts=108,requested_seconds=.25,timestamp_seconds=8/30)
    verify_source_mapping(v,[f],np.array([100,108,116,124]))


def test_feature_width_from_real_accepted_outputs(tmp_path):
    """接口：审计实际输出维度，不假设旧768或当前1408。"""
    durable(tmp_path/'workers/coarse_0.json',{'acceptance':{'status':'passed','feature_dim':1234}})
    assert actual_feature_dim(tmp_path)==1234
    durable(tmp_path/'workers/coarse_1.json',{'acceptance':{'status':'passed','feature_dim':111}})
    with pytest.raises(ValueError):actual_feature_dim(tmp_path)
