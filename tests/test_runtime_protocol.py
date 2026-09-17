from fractions import Fraction
import pytest
from videoqa_runtime.protocol import options_text, parse_answer, question_text
from videoqa_runtime.video import candidate_rows, uniform_selection


def test_vfr_grid_uses_actual_pts_with_nonzero_origin():
    # 0.0, 0.1, 0.26, 0.9, 1.27, 2.26 seconds: nominal-FPS indexing would be wrong.
    rows = list(candidate_rows([1000, 1100, 1260, 1900, 2270, 3260], Fraction(1, 1000), 1000))
    assert [r['source_frame_index'] for r in rows] == [2, 4, 5]
    assert [r['requested_seconds'] for r in rows] == [0.25, 1.25, 2.25]
    assert [r['timestamp_seconds'] for r in rows] == [0.26, 1.27, 2.26]


def test_slow_source_does_not_repeat_a_frame_across_grid_points():
    rows = list(candidate_rows([0, 250, 450], Fraction(1, 100), 0))
    assert [r['source_pts'] for r in rows] == [250, 450]
    assert len({r['source_frame_index'] for r in rows}) == 2


@pytest.mark.parametrize('pts', [[0, None], [0, 5, 5], [0, 5, 4]])
def test_bad_pts_stop_the_run(pts):
    with pytest.raises(ValueError, match='PTS'):
        list(candidate_rows(pts, Fraction(1, 1), 0))


def test_uniform_budget_and_round_to_even_ties():
    assert uniform_selection(list(range(6)), 3) == [0, 2, 5]
    result = uniform_selection(list(range(31)), 16)
    assert result == list(range(0, 31, 2))
    # 2026-09-17 修订：16帧预算为最大上限——15个候选返回全部15帧，仅空候选报错
    assert uniform_selection(list(range(15))) == list(range(15))
    with pytest.raises(ValueError):
        uniform_selection([])


def test_options_are_not_duplicated_or_reordered():
    assert options_text(['A. Cat', 'B. Dog', 'C. Bird', 'D. Fish']) == 'A. Cat\nB. Dog\nC. Bird\nD. Fish'
    with pytest.raises(ValueError):
        options_text(['B. Cat', 'A. Dog', 'C. Bird', 'D. Fish'])


@pytest.mark.parametrize('text,answer', [('A', 'A'), ('(B)', 'B'), ('Answer: C.', 'C'), ('A or B', None), ('A kangaroo', None), ('The answer is probably D', None), ('', None)])
def test_answer_parser_is_unambiguous(text, answer):
    assert parse_answer(text) == answer


def test_shared_prompt_does_not_claim_every_selector_is_uniform():
    text = question_text('What is shown?', ['Cat', 'Dog', 'Bird', 'Fish'], 20, list(range(16)))
    assert 'uniformly' not in text and text.count('<image>') == 1
    with pytest.raises(ValueError):
        question_text('What is shown?', ['Cat', 'Dog', 'Bird', 'Fish'], 20, [0] * 16)
