"""同位文字变化的边界与原框约束测试，全部零模型。"""
import pytest
from videoqa_methods.spatial_text_redundancy import layout_state, box_iou, character_change, layout_change


def field(text, box=(0, 0, .5, .5)):
    return ((box, text),)


def test_no_corresponding_text_preserves_visual_similarity():
    assert layout_change((), field('7')) == layout_change(field('7'), ()) == 0
    assert layout_change(field('7'), field('20', (.6, .6, 1, 1))) == 0


def test_same_field_different_values_have_bounded_symmetric_change():
    assert layout_change(field('5:59'), field('6:00')) == .75
    assert layout_change(field('6:00'), field('5:59')) == .75
    assert layout_change(field('7:30'), field('4:30')) == .25
    assert layout_change(field('10:00'), field('10:00')) == 0


def test_partial_box_overlap_attenuates_single_character_noise():
    assert box_iou((0, 0, .5, .5), (.25, 0, .75, .5)) == pytest.approx(1 / 3)
    assert layout_change(field('7'), field('0', (.25, 0, .75, .5))) == pytest.approx(1 / 3)


def test_shared_endpoint_has_zero_area_correspondence():
    assert box_iou((0, 0, .5, .5), (.5, 0, 1, .5)) == 0


def test_identical_overlapping_fields_never_have_self_change():
    fields = (((0, 0, .5, .5), 'alpha'), ((.25, 0, .75, .5), 'beta'))
    assert layout_change(fields, fields) == 0
    assert layout_change(fields, fields[::-1]) == 0


def test_geometry_not_text_difference_decides_correspondence():
    left = (((0, 0, .5, .5), 'alpha'), ((.25, 0, .75, .5), 'beta'))
    right = (((0, 0, .5, .5), 'alpha'), ((.25, 0, .75, .5), 'betx'))
    assert layout_change(left, right) == .25
    assert layout_change(right, left) == .25


def test_ambiguous_duplicate_boxes_do_not_reward_unresolved_text():
    left = (((0, 0, .5, .5), 'alpha'), ((0, 0, .5, .5), 'beta'))
    assert layout_change(left, field('other')) == 0


def test_edit_distance_keeps_word_order_and_normalizes_changes():
    assert character_change('abc', 'acb') == pytest.approx(2 / 3)
    assert character_change('a', 'abc') == character_change('abc', 'a') == pytest.approx(2 / 3)


def test_source_canvas_and_legacy_confidence_are_explicit():
    regions = [dict(box=[[10, 10], [60, 10], [60, 60], [10, 60]], text='  STATE  A ', score=.9),
               dict(box=[[10, 10], [60, 10], [60, 60], [10, 60]], text='noise', score=.49)]
    assert layout_state(regions, [100, 100, 3]) == (((.1, .1, .6, .6), 'state a'),)
    assert layout_state(regions, [200, 200, 3])[0][0] == (.05, .05, .3, .3)


@pytest.mark.parametrize('change', ['bad_score', 'bad_box', 'outside_canvas'])
def test_invalid_ocr_geometry_or_confidence_stops(change):
    region = dict(box=[[10, 10], [60, 10], [60, 60], [10, 60]], text='value', score=.9)
    if change == 'bad_score': region['score'] = float('nan')
    if change == 'bad_box': region['box'] = [[1, 2]]
    if change == 'outside_canvas': region['box'][0] = [-1, -1]
    with pytest.raises(ValueError): layout_state([region], [100, 100, 3])
