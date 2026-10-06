"""空间关注仅改变观测接口、数值和零质量边界测试，不消耗研究模型预算。"""
import numpy as np
import pytest
import torch
from videoqa_audit.blip_spatial_state import capture_cross_attention, compare_pair


class Attention(torch.nn.Module):
    def forward(self, hidden, mask=None, head=None, encoder=None, encoder_mask=None, cache=None, output_attentions=False):
        probs = hidden.softmax(-1)
        context = probs @ torch.tensor([[1.0], [2.0]])
        return (context, probs, ("cache",)) if output_attentions else (context, ("cache",))


def test_hook_preserves_output_and_gradient_and_restores_interface():
    module = Attention()
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    normal = module(x)
    with capture_cross_attention(module) as captured:
        observed = module(x, None, None, None, None, None, False)
        gradient, = torch.autograd.grad(observed[0].sum(), captured["attention"])
        assert torch.equal(normal[0], observed[0]) and observed[1] == normal[1]
        assert torch.equal(gradient, torch.tensor([[1.0, 2.0]]))
    assert len(module(x)) == 2 and not module._forward_hooks and not module._forward_pre_hooks


def test_hook_restored_after_exception():
    module = Attention()
    with pytest.raises(RuntimeError):
        with capture_cross_attention(module):
            raise RuntimeError("abort")
    assert not module._forward_hooks and not module._forward_pre_hooks


def test_query_weight_local_change_is_not_global_cls_change():
    a = np.tile([1.0, 0.0], (576, 1))
    b = a.copy()
    b[0] = [0.0, 1.0]
    cam = np.zeros(576)
    cam[0] = 1
    value = compare_pair(np.array([1.0, 0.0]), np.array([1.0, 0.0]), a, b, cam, cam)
    assert value["cls_difference"] == 0
    assert value["uniform_patch_difference"] == pytest.approx(1/576)
    assert value["query_patch_difference"] == 1


def test_zero_gradcam_is_reported_invalid_without_uniform_fallback():
    a = np.tile([1.0, 0.0], (576, 1))
    value = compare_pair(np.array([1.0, 0.0]), np.array([1.0, 0.0]), a, a, np.zeros(576), np.ones(576))
    assert not value["spatial_valid"] and value["query_patch_difference"] is None


def test_negative_gradcam_rejected():
    a = np.tile([1.0, 0.0], (576, 1))
    with pytest.raises(ValueError, match="spatial map"):
        compare_pair(np.array([1.0, 0.0]), np.array([1.0, 0.0]), a, a, -np.ones(576), np.ones(576))


def load_runner():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("spatial_runner", Path(__file__).resolve().parents[1] / "scripts/probe_blip_spatial_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_call_restores_without_forward_and_changed_tokens_rejected(tmp_path, monkeypatch):
    import json
    module = load_runner()
    (tmp_path / "protocol.json").write_text(json.dumps(dict(real_forward_limit=2, synthetic_forward_limit=2)))
    count = []
    monkeypatch.setattr(module, "spatial_batch", lambda *_a, **_k: (count.append(1) or dict(scores=np.ones(2)), {}))
    encoded = dict(input_ids=torch.tensor([[1, 2]]))
    module.invoke(tmp_path, "q", "control", None, encoded, [None, None])
    module.invoke(tmp_path, "q", "control", None, encoded, [None, None])
    assert count == [1]
    with pytest.raises(ValueError, match="repetition forbidden"):
        module.invoke(tmp_path, "q", "control", None, dict(input_ids=torch.tensor([[1, 3]])), [None, None])
    with pytest.raises(ValueError, match="budget exhausted"):
        module.invoke(tmp_path, "q2", "control", None, encoded, [None])


def test_uncertain_started_call_never_retried(tmp_path, monkeypatch):
    import json
    module = load_runner()
    (tmp_path / "protocol.json").write_text(json.dumps(dict(real_forward_limit=2, synthetic_forward_limit=2)))
    def fail(*_args, **_kwargs):
        raise RuntimeError("unknown forward state")
    monkeypatch.setattr(module, "spatial_batch", fail)
    encoded = dict(input_ids=torch.tensor([[1, 2]]))
    with pytest.raises(RuntimeError):
        module.invoke(tmp_path, "q", "control", None, encoded, [None])
    with pytest.raises(ValueError, match="repetition forbidden"):
        module.invoke(tmp_path, "q", "control", None, encoded, [None])
