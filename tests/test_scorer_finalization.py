"""评分器收尾回归：引用成本隔离与恢复字节变化必须能够被发现。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

spec = importlib.util.spec_from_file_location('scorer_finalization',
    Path(__file__).resolve().parents[1] / 'scripts/finalize_scorer_comparison.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_generation_identity_detects_mutation_and_new_file(tmp_path):
    """接口：恢复前后必须同时核对文件集合与内容，不能只比较生成数量。"""
    path = tmp_path / 'qa/qwen32/raw_generations/m/001-1.json'
    path.parent.mkdir(parents=True); path.write_text('{}')
    before = module.generation_identity(tmp_path)
    assert before == module.generation_identity(tmp_path)
    path.write_text('{"changed": true}')
    assert before != module.generation_identity(tmp_path)
    before = module.generation_identity(tmp_path)
    path.with_name('001-2.json').write_text('{}')
    assert before != module.generation_identity(tmp_path)


def test_only_primary_has_physical_cost_but_all_results_have_task_counts(tmp_path):
    """接口：共享回答计入方法正确数，但不能复制主调用的解码和QA成本。"""
    base = dict(method='m', correct=True, timing_kind='physical_primary',
                answer=dict(framework_metric=dict(task_category='Action Recognition'),
                    observed=dict(qa_seconds=2., peak_allocated_gib=4.)),
                timings=dict(final_decode_and_identity_seconds=1., cached_input_to_answer_seconds=3.))
    folder = tmp_path / 'qa/qwen32/results/m'; folder.mkdir(parents=True)
    (folder / '001-1.json').write_text(json.dumps(base))
    base['timing_kind'] = 'shared_reference'
    (folder / '001-2.json').write_text(json.dumps(base))
    value = module.physical_costs(tmp_path, 'qwen32')
    assert value['timing_kind_counts']['m'] == dict(physical_primary=1, shared_reference=1)
    assert value['official_task_categories']['Action Recognition']['m'] == dict(n=2, correct=2)
    assert value['physical_performance']['m']['qa_seconds']['mean'] == 2.


def test_qwen_preview_reverses_official_temporal_and_spatial_packing():
    """接口：可见性预览不能交换时间、通道、merge块或patch内部像素。"""
    path = Path(__file__).resolve().parents[1] / 'scripts/export_scorer_input_views.py'
    spec = importlib.util.spec_from_file_location('scorer_preview', path)
    preview = importlib.util.module_from_spec(spec); spec.loader.exec_module(preview)
    frames = np.arange(2 * 3 * 28 * 56).reshape(2, 3, 28, 56)
    flat = frames.reshape(1, 2, 3, 1, 2, 14, 2, 2, 14).transpose(
        0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(8, 3 * 2 * 14 * 14)
    processor = SimpleNamespace(patch_size=14, merge_size=2, temporal_patch_size=2)
    assert np.array_equal(frames, preview.unpack_video(flat, [1, 2, 4], processor))


def test_task_comparison_restores_both_scorers_and_uniform(tmp_path, monkeypatch):
    """接口：题型统计按四格题号复算，不能把S2新增正确误归为S1正确。"""
    import videoqa_full.state
    monkeypatch.setattr(videoqa_full.state, 'rows', lambda: [
        dict(question_id='q1', task_type='object'), dict(question_id='q2', task_type='count')])
    cells = dict(both_correct=[], left_only=['q2'], right_only=['q1'], both_wrong=[])
    paired = {'S2_minus_S1__' + m: dict(all2700=dict(cells=cells))
              for m in ('BASE-BLIP-TopK', 'RD-1.2-exp07', 'RD-1.2-exp12')}
    paired['S2__RD-1.2-exp07_minus_BASE-Uniform'] = dict(all2700=dict(cells=cells))
    (tmp_path / 'comparison.json').write_text(json.dumps(dict(profiles=dict(qwen32=dict(paired=paired)))))
    values = module.task_comparison(tmp_path)['qwen32']
    assert values['S1']['object']['RD-1.2-exp07']['correct'] == 1
    assert values['S2']['object']['RD-1.2-exp07']['correct'] == 0
    assert values['S2']['count']['RD-1.2-exp07']['correct'] == 1
    assert values['S1']['object']['BASE-Uniform'] == values['S2']['object']['BASE-Uniform']
