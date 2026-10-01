"""实际补丁布局校验器的空间/时间还原测试，不加载新模型。"""
import numpy as np
import pytest
from videoqa_audit.framegroup_embedding import unpack_pixels


def test_inverse_patch_packing_keeps_temporal_and_spatial_order():
    images=np.arange(4*8*8*3,dtype=np.float32).reshape(4,8,8,3)
    packed=images.reshape(2,2,2,2,2,2,2,2,3).transpose(0,2,5,3,6,8,1,4,7).reshape(2*4*4,3*2*2*2)
    actual=unpack_pixels(packed,[[2,4,4]],patch=2,merge=2,temporal=2)
    assert np.array_equal(actual,images)


def test_invalid_grid_does_not_silently_reshape():
    with pytest.raises(ValueError):unpack_pixels(np.zeros((2,3)),[[1,3,3]])
