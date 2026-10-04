"""固定裁剪坐标测试，不运行图像模型。"""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('spatial',Path(__file__).parents[1]/'scripts/probe_blip_spatial_scale.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_fixed_even_boxes():
    """偶数尺寸严格使用半宽半高，中心窗不依赖目标物体。"""
    assert module.crop_boxes(1280,720)==[(0,0,640,360),(640,0,1280,360),
                                      (0,360,640,720),(640,360,1280,720),(320,180,960,540)]


def test_odd_boxes_stay_in_bounds():
    """奇数尺寸向下取整，各窗面积一致且不越界。"""
    boxes=module.crop_boxes(641,361)
    assert len(boxes)==5
    assert all(0<=x<r<=641 and 0<=y<b<=361 and r-x==320 and b-y==180 for x,y,r,b in boxes)


def test_tiny_image_rejected():
    """不允许生成零面积裁剪。"""
    with pytest.raises(ValueError):module.crop_boxes(1,20)
