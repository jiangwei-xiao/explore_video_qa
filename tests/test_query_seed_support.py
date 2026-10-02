"""诊断权重的代数和边界验收，不依赖模型输出或正确答案。"""
import numpy as np
import pytest
from videoqa_audit.query_seed_support import query_supported_seed_weights

PLAN={'regions':[{'region_id':0,'seed_indices':[0,2]},{'region_id':1,'seed_indices':[1,3]}]}


def test_identical_views_keep_original_relative_weights():
    scores=[.8,.3,.2,.6]
    weights,_=query_supported_seed_weights(PLAN,[scores]*5)
    np.testing.assert_array_equal(weights,[1.,.5,.25,1.])


def test_max_operator_commutes_without_query_calibration():
    scores=np.array([[.8,.3,.2,.6],[.1,.02,.9,.08]])
    weights,_=query_supported_seed_weights(PLAN,scores)
    similarities=np.array([[.9,.1],[.4,.7],[.2,.8]])
    normalized=scores[:,[0,2]]/scores[:,[0,2]].max(axis=1,keepdims=True)
    direct=np.max(similarities[:,None,:]*normalized[None,:,:],axis=(1,2))
    fused=np.max(similarities*np.array(weights)[[0,2]],axis=1)
    np.testing.assert_array_equal(direct,fused)


def test_zero_query_does_not_override_nonzero_evidence():
    scores=[.8,.3,.2,.6]
    weights,_=query_supported_seed_weights(PLAN,[scores,[0]*4])
    np.testing.assert_array_equal(weights,[1.,.5,.25,1.])
    weights,trace=query_supported_seed_weights(PLAN,[[0]*4])
    assert weights==[1.]*4 and all(t['all_zero_fallback'] for t in trace.values())


def test_reject_overlapping_seeds_and_invalid_scores():
    with pytest.raises(ValueError):
        query_supported_seed_weights({'regions':[{'region_id':0,'seed_indices':[0]},{'region_id':1,'seed_indices':[0]}]},[[.2]])
    with pytest.raises(ValueError):
        query_supported_seed_weights(PLAN,[[np.nan]*4])
