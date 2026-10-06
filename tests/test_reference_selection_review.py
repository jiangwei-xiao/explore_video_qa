"""参考对照源身份与双向案例冻结的必要边界测试。"""
import pytest
from videoqa_audit.reference_selection_review import compare_source_frames, freeze_reference_panel


def selection(pts, path="v.mp4"):
    return dict(video=dict(path=path, time_base="1/10", start_pts=5),
                selected_frames=[dict(source_frame_index=i, source_pts=p) for i, p in enumerate(pts)])


def test_temporal_neighbour_is_not_source_identity():
    record = compare_source_frames(selection([5, 15]), selection([6, 16]))
    assert record["exact_shared_sources"] == 0
    assert record["right_temporally_near_left_half_second"] == 2
    assert record["left"]["span_seconds"] == 1


def test_different_video_cannot_be_matched():
    with pytest.raises(ValueError, match="Video identity"):
        compare_source_frames(selection([5]), selection([5], "other.mp4"))


def test_unordered_sources_rejected():
    with pytest.raises(ValueError, match="Unordered"):
        compare_source_frames(selection([15, 5]), selection([5]))


def test_panel_outside_development_and_empty_slots():
    data = [dict(question_id="a", video_id="v1", stratum="short"),
            dict(question_id="b", video_id="v2", stratum="short"),
            dict(question_id="c", video_id="v3", stratum="long")]
    records = {r["question_id"]: dict(wfs_only_both_rd_wrong=True,
                                    both_rd_correct_wfs_wrong=False) for r in data}
    panel = freeze_reference_panel(data, records, dict(questions=[]), {"v1"})
    assert [q["question_id"] for q in panel["questions"]] == ["b"]
    assert sum(s["question_id"] is None for s in panel["additional_slots"]) == 3


def test_preexisting_panel_video_excluded():
    data = [dict(question_id=q, video_id="same", stratum="short") for q in ("a", "b")]
    records = {r["question_id"]: dict(wfs_only_both_rd_wrong=True,
                                    both_rd_correct_wfs_wrong=False) for r in data}
    previous = dict(questions=[dict(question_id="a", slot="old")])
    panel = freeze_reference_panel(data, records, previous, set())
    assert len(panel["questions"]) == 1
    assert all(s["question_id"] is None for s in panel["additional_slots"])


@pytest.mark.parametrize("prefix_missing", [False, True])
def test_single_witness_reachability_does_not_imply_group_reachability(monkeypatch, tmp_path, prefix_missing):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/trace_reference_review_groups.py"
    spec = importlib.util.spec_from_file_location("reference_groups", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pool = dict(initial_count=6, candidates=[dict(source_frame_index=i, source_pts=i) for i in range(6)])
    trace = dict(region_budgets={"0": 2, "1": 2}, regions=[
        dict(region_id=0, members=[0, 1, 2], reference_coarse_indices=[0, 1, 2], prefix=[0, 1, 2]),
        dict(region_id=1, members=[3, 4, 5], reference_coarse_indices=[3, 4, 5],
             prefix=[3, 4] if prefix_missing else [3, 4, 5]),
    ])
    monkeypatch.setattr(module, "load_pair", lambda *_: (pool, trace, trace))
    monkeypatch.setattr(module, "durable", lambda *_: None)
    result = module.trace_groups(tmp_path, [dict(question_id="q", coarse_indices=[2, 5])])[0]["exp12"]
    assert not result["group_reachable_by_quota_only"]
    assert result["group_in_prefix"] is not prefix_missing
    if not prefix_missing:
        assert all(p["state"] == "quota_reachable" for p in result["paths"]["paths"])
        assert sum(result["minimum_quotas"].values()) == 6


def test_flat_canvas_flag_does_not_flag_black_text_or_dark_object():
    import importlib.util
    from pathlib import Path
    from PIL import Image
    path = Path(__file__).resolve().parents[1] / "scripts/trace_reference_review_groups.py"
    spec = importlib.util.spec_from_file_location("reference_flatness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = Image.new("RGB", (10, 10), (10, 10, 10))
    assert module.measure_flatness(image)["nearly_constant_canvas"]
    image.putpixel((5, 5), (15, 15, 15))
    assert not module.measure_flatness(image)["nearly_constant_canvas"]
    image = Image.new("RGB", (10, 10), (0, 0, 0))
    image.putpixel((5, 5), (255, 255, 255))
    assert not module.measure_flatness(image)["nearly_constant_canvas"]
