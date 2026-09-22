"""正式软折中入口连接测试；选择边界由局部验证测试覆盖。"""
from videoqa_audit.density_local_validation import supported_selection
from videoqa_density_soft import density_pipeline, density_run, density_report
from videoqa_methods import density_pipeline as original_pipeline
from videoqa_methods import density_run as original_run


def test_selector_is_shared_without_mutating_original():
    """正式和离线使用同一选择实现，旧入口不被全局替换。"""
    assert density_pipeline.select_joint is supported_selection
    assert density_run.select_joint is supported_selection
    assert original_pipeline.select_joint is not supported_selection
    assert original_run.select_joint is not supported_selection


def test_raw_scan_decode_and_fixed_config_reused():
    """唯一改变在选择层；候选构造和配置沿用旧版。"""
    assert density_pipeline.scan_original is original_pipeline.scan_original
    assert density_pipeline.decode_regions is original_pipeline.decode_regions
    assert density_run.CONFIG == original_run.CONFIG
    assert 'Density_v1' in density_run.HISTORY_PATHS
    assert 'Density_v1' not in original_run.HISTORY_PATHS


def test_reporting_statistics_reused():
    """确认统计接口的固定分母/置信区间行为。"""
    result = density_report.accuracy([{'correct': True}, {'correct': False}])
    assert result['n'] == 2 and result['correct'] == 1
    assert result['wilson95'][0] < .5 < result['wilson95'][1]
