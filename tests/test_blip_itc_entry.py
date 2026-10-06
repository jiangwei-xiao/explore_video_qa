"""官方ITC入口排序与缓存投影独立测试，不调用研究模型。"""
import numpy as np
import pytest
from videoqa_audit.blip_itc_entry import rank_itc_candidates,score_itc_entry


def test_itc_negative_cosines_and_stable_time_ties():
    c=[dict(source_pts=30,candidate_index=0),dict(source_pts=20,candidate_index=1),dict(source_pts=10,candidate_index=2)]
    assert rank_itc_candidates(c,[-.2,.3,.3],2)==[2,1]
    assert rank_itc_candidates(c,[-.2,.3,.3],8)==[2,1,0]


@pytest.mark.parametrize('values',([2.,0.],[float('nan'),0.],[0.]))
def test_itc_invalid_scores_rejected(values):
    with pytest.raises(ValueError):
        rank_itc_candidates([dict(source_pts=0,candidate_index=0)]*2,values,1)


def test_text_cosine_uses_both_trained_projections_not_frame_frame_similarity(monkeypatch):
    from videoqa_audit import blip_projection
    g=np.zeros((2,256),dtype=np.float32);g[0,0]=1;g[1,1]=1
    monkeypatch.setattr(blip_projection,'recover_projected_features',lambda f,p:(g,np.ones(2)))
    t=np.zeros(256,dtype=np.float32);t[1]=1
    scores,scale=score_itc_entry(np.zeros((2,768)),t,{})
    assert np.array_equal(scores,[0.,1.])
    assert np.array_equal(scale,np.ones(2))


def test_itc_text_projection_must_be_normalized(monkeypatch):
    from videoqa_audit import blip_projection
    monkeypatch.setattr(blip_projection,'recover_projected_features',lambda f,p:(np.ones((1,256)),np.ones(1)))
    with pytest.raises(ValueError,match='text'):
        score_itc_entry(np.zeros((1,768)),np.ones(256),{})
