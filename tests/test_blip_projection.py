"""LayerNorm尺度恢复的代数与失败边界，真实模型一致性另行测试。"""
import numpy as np
import pytest
from videoqa_audit.blip_projection import recover_projected_features


def fixture():
    rng=np.random.default_rng(2027);z=rng.normal(size=(20,24));z-=z.mean(axis=1,keepdims=True);z/=np.sqrt((z*z).mean(axis=1,keepdims=True)+1e-5)
    p=dict(gamma=rng.uniform(.4,.8,24),beta=rng.uniform(-.2,-.1,24),weight=rng.normal(size=(8,24)),bias=rng.normal(size=8))
    raw=z*p['gamma']+p['beta'];f=raw/np.linalg.norm(raw,axis=1,keepdims=True)
    direct=raw@p['weight'].T+p['bias'];direct/=np.linalg.norm(direct,axis=1,keepdims=True)
    return f,p,raw,direct


def test_recovery_includes_nonzero_projection_bias():
    f,p,raw,expected=fixture();actual,scales=recover_projected_features(f,p)
    assert np.max(np.abs(actual-expected))<1e-7
    assert np.max(np.abs(scales-np.linalg.norm(raw,axis=1)))<1e-12


def test_float32_cache_recovery_is_bounded():
    f,p,raw,expected=fixture();actual,_=recover_projected_features(f.astype(np.float32),p)
    assert np.max(np.abs(actual-expected))<1e-6


def test_zero_bias_sum_is_not_silently_guessed():
    f,p,_,_=fixture();p['beta']=np.zeros(24)
    with pytest.raises(ValueError):recover_projected_features(f,p)


def test_zero_layernorm_scale_rejected():
    f,p,_,_=fixture();p['gamma'][0]=0
    with pytest.raises(ValueError):recover_projected_features(f,p)


def test_non_normalized_cache_rejected():
    f,p,_,_=fixture()
    with pytest.raises(ValueError):recover_projected_features(2*f,p)
