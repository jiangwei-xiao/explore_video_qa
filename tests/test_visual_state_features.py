"""exp24接口、单帧时间隔离、均值数值与实际竞争快照测试，不运行研究模型。"""
import numpy as np
import pytest
import torch
from videoqa_methods.visual_state_features import pool_visual_tokens, competition_snapshot


def test_pooling_keeps_video_boundaries_and_uses_float32_mean():
    tokens = torch.tensor([[2., 0.], [2., 0.], [0., 3.]], dtype=torch.bfloat16)
    f, lengths = pool_visual_tokens(tokens, [[1, 2, 4], [1, 2, 2]])
    assert lengths == [2, 1]
    assert f.dtype == torch.float32
    assert torch.equal(f, torch.eye(2))


@pytest.mark.parametrize('grid', ([[2, 2, 2]], [[1, 3, 2]], [[1, 0, 2]], [[1, 2]]))
def test_non_single_frame_and_invalid_spatial_grid_rejected(grid):
    with pytest.raises(ValueError):
        pool_visual_tokens(torch.ones(1, 2), grid)


def test_visual_token_population_and_zero_state_rejected():
    with pytest.raises(ValueError, match='population'):
        pool_visual_tokens(torch.ones(2, 2), [[1, 2, 2]])
    with pytest.raises(ValueError, match='mean'):
        pool_visual_tokens(torch.zeros(1, 2), [[1, 2, 2]])


def test_competition_snapshot_only_compares_same_frozen_context():
    pool = dict(plan=dict(regions=[dict(region_id=0, seed_indices=[0])]), initial_scores=[1., .5, .5])
    f = {0: np.array([1., 0.]), 1: np.array([.8, .6]), 2: np.array([.99, np.sqrt(1-.99**2)])}
    result = competition_snapshot(pool, dict(region_id=0), [0], 1, 2, f)
    assert result['witness_ahead']
    assert result['selected_before'] == [0]
    assert result['witness']['support'] == pytest.approx(.8)
    assert result['witness']['local_score'] == pytest.approx(.8*np.sqrt(.2))


def test_encoder_rejects_batch_over_eight_before_loading_cuda():
    from videoqa_methods.visual_state_features import VisualStateEncoder
    encoder = VisualStateEncoder.__new__(VisualStateEncoder)
    with pytest.raises(ValueError, match='batch'):
        encoder.encode([None]*9)


def test_processor_media_call_has_explicit_images_and_independent_video_axis():
    from videoqa_methods.visual_state_features import preprocess_independent_frames
    class Processor:
        def preprocess(self, images, videos, size, return_tensors):
            assert images is None and return_tensors == 'pt'
            assert size == {'shortest_edge':100352,'longest_edge':602112}
            assert len(videos) == 2 and all(v.shape == (1, 4, 6, 3) for v in videos)
            assert np.all(videos[0] == 0) and np.all(videos[1] == 255)
            return {'valid': True}
    assert preprocess_independent_frames(Processor(), [np.zeros((4,6,3)),np.full((4,6,3),255)]) == {'valid': True}


def test_official_processor_actual_grid_matches_declared_pixels():
    from transformers import AutoProcessor
    from pathlib import Path
    from videoqa_methods.visual_state_features import preprocess_independent_frames
    model = Path('/home/models/Qwen/Qwen2.5-VL-7B-Instruct')
    if not model.exists():
        pytest.skip('Official local processor unavailable')
    processor = AutoProcessor.from_pretrained(model, local_files_only=True, use_fast=False,
        min_pixels=100352,max_pixels=602112).image_processor
    data = preprocess_independent_frames(processor, [np.zeros((720,1280,3),dtype=np.uint8),np.ones((1280,720,3),dtype=np.uint8)])
    grid = data['video_grid_thw']
    assert torch.all(grid[:,0]==1)
    assert torch.all((grid[:,1]*grid[:,2]*196>=100352)&(grid[:,1]*grid[:,2]*196<=602112))


def test_original_features_reproduce_exp12_and_fixed_parent_allocation():
    from test_prefix_coverage_budget import fixture
    from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
    from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
    from videoqa_methods.visual_state_features import select_with_state_features
    args = fixture()
    c, f, plan, video, cfg, quotas, scores = args
    pool = dict(candidates=c, plan=plan, video=video, initial_scores=scores, initial_count=len(scores))
    _, prefix = select_prefix_coverage(*args)
    expected, parent = select_relevance_coverage(pool, f, prefix)
    actual, trace = select_with_state_features(pool, f, cfg, parent)
    assert actual == expected == trace['fixed_parent_quota_indices']
    assert trace['region_budgets'] == parent['region_budgets']
    assert trace['objective'] == parent['objective']


def load_visual_runner():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('visual_state_runner', Path(__file__).resolve().parents[1]/'scripts/run_visual_state_exploration.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visual_forward_restores_without_repeat_and_changed_review_rejected(tmp_path, monkeypatch):
    from PIL import Image
    import hashlib
    module = load_visual_runner()
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'labels_review.json').write_text('{}')
    image = Image.new('RGB',(8,8),'white')
    calls = []
    def fake(images, encoder):
        calls.append(1)
        return np.array([[1.,0.]],dtype=np.float32), dict(rgb_sha256=[hashlib.sha256(np.asarray(image).tobytes()).hexdigest()])
    monkeypatch.setattr(module, 'encode_visual_states', fake)
    module.invoke(tmp_path,'q',[image],None)
    module.invoke(tmp_path,'q',[image],None)
    assert calls == [1]
    (tmp_path/'labels_review.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='repetition forbidden'):
        module.invoke(tmp_path,'q',[image],None)


def test_uncertain_visual_forward_is_never_repeated(tmp_path, monkeypatch):
    from PIL import Image
    module = load_visual_runner()
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'labels_review.json').write_text('{}')
    def fail(*args):
        raise RuntimeError('unknown visual return')
    monkeypatch.setattr(module,'encode_visual_states',fail)
    with pytest.raises(RuntimeError):
        module.invoke(tmp_path,'q',[Image.new('RGB',(8,8))],None)
    with pytest.raises(ValueError, match='repetition forbidden'):
        module.invoke(tmp_path,'q',[Image.new('RGB',(8,8))],None)
