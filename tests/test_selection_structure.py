"""同粗池对照的真实参考CPU验收；不加载视频、权重或历史答案。"""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from videoqa_audit import selection_structure as study


@pytest.fixture
def core_path():
    """接口：允许隔离CPU环境读取同SHA的临时参考源码，缺失时明确跳过。"""
    path = Path(os.environ.get('WFS_REFERENCE_TEST_PATH', study.REFERENCE))
    if not path.is_file():
        pytest.skip('Pinned official WFS reference is not installed in this checkout')
    study.reference_core(path)
    return path


def inputs(n=73):
    """接口：生成含超32位PTS及不等时间间隔的合法合成粗池。"""
    rng = np.random.default_rng(2027)
    f = rng.normal(size=(n, 19)).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    start = 2**34
    ticks = np.cumsum(rng.integers(801, 1199, n))
    rows = [dict(candidate_index=i, source_pts=int(start+t), source_frame_index=i*29,
                 timestamp_seconds=float(t/1000)) for i, t in enumerate(ticks)]
    pool = dict(initial_count=n, initial_scores=rng.random(n).tolist(), candidates=rows,
                video=dict(start_pts=start, time_base='1/1000', duration_fraction=str(n*2)))
    return pool, f


@pytest.mark.parametrize('method', study.CONTROLS)
@pytest.mark.parametrize('n,k', [(1, 8), (7, 8), (8, 8), (11, 8), (14, 32), (73, 8), (73, 32)])
def test_actual_reference_exact(method, n, k, core_path):
    """接口：逐项和未经观察包装的官方调用比较，不以自写算法作正确性来源。"""
    pool, features = inputs(n)
    before = copy.deepcopy(pool); f_before = features.copy()
    got, trace = study.select_coarse_control(pool, features, k, method, _reference=core_path)
    core = study.reference_core(core_path); scores = np.array(pool['initial_scores'])
    if method == study.GLOBAL:
        expected = core.WFSFrameSelector(.5).select_from_segment((0, n), min(k, n), 2*scores-1, features)
    else:
        expected = core.WFS(core.WFSConfig(**study.PAPER_CONFIG)).select_keyframes(
            scores, k, core.compute_dwt_level(n, drift=3), core.compute_min_peak_distance(n), features)
        assert trace['config'] == study.PAPER_CONFIG
    assert got == sorted(expected)
    assert len(got) == len(set(got)) == min(n, k)
    assert pool == before
    np.testing.assert_array_equal(features, f_before)
    assert json.loads(json.dumps(trace,allow_nan=False))['selected_indices']==got


def test_global_uses_negative_cosine_and_original_ties(core_path):
    """接口：反向特征必须得到负余弦奖励；并列保留官方原候选顺序。"""
    pool, _ = inputs(4)
    features = np.array([[1, 0], [1, 0], [-1, 0], [0, 1]], dtype=np.float32)
    pool['initial_scores'] = [.9, .9, .5, .5]
    _, trace = study.select_coarse_control(pool, features, 2, study.GLOBAL, _reference=core_path)
    assert trace['selection_order'] == [0, 2]
    assert trace['steps'][1]['maximum_selected_cosine'] == -1.
    assert trace['steps'][1]['relevance'] == 0.


@pytest.mark.parametrize('score', [0., .5, 1.])
def test_native_constant_score_branch(score, core_path):
    """接口：等分时也保留小波浮点边界效应，不擅自将细小峰值抹成零。"""
    pool, f = inputs(73); pool['initial_scores'] = [score]*73
    got, trace = study.select_coarse_control(pool, f, 8, study.WFS, _reference=core_path)
    core = study.reference_core(core_path)
    expected = core.WFS(core.WFSConfig(**study.PAPER_CONFIG)).select_keyframes(
        np.full(73, score), 8, core.compute_dwt_level(73), core.compute_min_peak_distance(73), f)
    assert got == expected
    if score == 0.:
        assert trace['fallback'] == 'official_uniform_topk'
    else:
        expected_branch = 'official_uniform_topk' if not trace['observed']['peaks'] else 'official_segment_mmr'
        assert trace['fallback'] == expected_branch


def test_defaults_are_not_paper_configuration(core_path):
    """接口：锁定容易混淆的参考默认参数差异。"""
    core = study.reference_core(core_path)
    assert core.WFSConfig().strictness_factor == 1.5
    assert study.PAPER_CONFIG['strictness_factor'] == 1.2
    assert study.PAPER_CONFIG['w_duration'] == .4


@pytest.mark.parametrize('damage', ['fine', 'score', 'feature', 'duplicate', 'clock', 'index'])
def test_bad_pool_rejected(damage):
    """接口：混池、特征错位、非有限数及源时钟错误必须在调用参考前拒绝。"""
    pool, f = inputs()
    if damage == 'fine': pool['initial_count'] -= 1
    if damage == 'score': pool['initial_scores'][0] = float('nan')
    if damage == 'feature': f = f[:-1]
    if damage == 'duplicate': pool['candidates'][1]['source_pts'] = pool['candidates'][0]['source_pts']
    if damage == 'clock': pool['candidates'][0]['timestamp_seconds'] += .01
    if damage == 'index': pool['candidates'][0]['candidate_index'] = 10
    with pytest.raises(ValueError): study.validate_coarse(pool, f, 8)


def test_reference_fingerprint_rejected(tmp_path):
    """接口：参考文件任何变化均拒绝，不能静默使用最新版。"""
    path = tmp_path/'core.py'; path.write_text('raise RuntimeError("must not import")')
    with pytest.raises(ValueError, match='fingerprint'): study.reference_core(path)
