"""双表征诊断的CPU几何边界测试；真实GPU重放另行验收。"""
import numpy as np
import pytest
from videoqa_audit.blip_joint_probe import geometry_summary,score_dual_embeddings


def test_geometry_excludes_self_and_preserves_negative_cosine():
    value=geometry_summary(np.array([[1,0],[0,1],[-1,0]],dtype=np.float32))
    assert value['pairs']==3 and value['cosine_quantiles'][0]==-1 and value['cosine_quantiles'][-1]==0


def test_geometry_rejects_unnormalized_input():
    with pytest.raises(ValueError):geometry_summary([[2,0],[0,1]])


def test_empty_batch_rejected_before_cuda():
    with pytest.raises(ValueError):score_dual_embeddings(None,None,[])
