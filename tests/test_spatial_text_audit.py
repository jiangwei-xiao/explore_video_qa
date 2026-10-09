"""字段贡献审计与人工擦除边界回归，全部使用合成CPU数组。"""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_spatial_text_pilot import matched_changes, remove_labeled_timer, audit_steps


def test_correspondence_audit_preserves_geometry_and_self_invariance():
    a = (((0, 0, .5, .5), 'alpha'), ((.25, 0, .75, .5), 'beta'))
    b = (((0, 0, .5, .5), 'alpha'), ((.25, 0, .75, .5), 'betx'))
    assert all(p['contribution'] == 0 for p in matched_changes(a, a[::-1]))
    result = matched_changes(a, b)
    assert result[0]['left_text'] == 'beta'
    assert result[0]['right_text'] == 'betx'
    assert result[0]['contribution'] == .25
    assert matched_changes((), b) == []


def test_ambiguous_spatial_fields_do_not_create_diagnostic_matches():
    a = (((0, 0, .5, .5), 'alpha'), ((0, 0, .5, .5), 'beta'))
    assert matched_changes(a, (((0, 0, .5, .5), 'other'),)) == []


def test_manual_erasure_is_limited_to_labeled_clock_and_does_not_mutate_input():
    state = [((.9, .04, .99, .09), '00:41'),
        ((.1, .04, .2, .09), '00:41'), ((.9, .04, .99, .09), 'banks')]
    masked, removed = remove_labeled_timer([state])
    assert len(removed) == 1 and removed[0]['layout_index'] == 0
    assert masked == [state[1:]]
    assert len(state) == 3


def test_competition_audit_replays_score_and_rejects_future_reference():
    pool = dict(candidates=[dict(source_pts=0), dict(source_pts=1)])
    f = np.asarray([[1., 0.], [.8, .6]], dtype=np.float32)
    states = [(((0, 0, .5, .5), '7'),), (((0, 0, .5, .5), '4'),)]
    step = dict(candidate_index=1, redundancy_reference=0, support=.9,
        redundancy=0., local_score=.9)
    trace = dict(regions=[dict(region_id=0, prefix=[0, 1], quota=2, local_steps=[step])])
    result = audit_steps(pool, f, states, trace)
    assert result[0]['prefix_position'] == 2 and result[0]['finally_selected']
    assert result[0]['dominating_fields'][0]['contribution'] == 1
    step['redundancy_reference'] = 1
    with pytest.raises(ValueError, match='not selected yet'):
        audit_steps(pool, f, states, trace)
