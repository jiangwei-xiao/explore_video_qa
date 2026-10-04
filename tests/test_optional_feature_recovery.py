"""特征批次恢复只使用完成资产，不触发未知前向重试。"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('feature_prep',Path(__file__).parents[1]/'scripts/prepare_optional_seed64_features.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def cached(tmp_path,status='completed'):
    """建立一条纯本地批次记录，不需要Torch或GPU。"""
    frames=[dict(source_pts=1,source_frame_index=1)]
    array=tmp_path/'batch_features/fine/q/0.npy'
    digest=module.save_array(array,np.eye(1,3,dtype=np.float32))
    record=dict(status=status,fingerprint='fp',gpu='0',frames=frames,feature_sha256=digest)
    module.durable(tmp_path/'calls/fine/q/0.json',record)
    return dict(selected_frames=frames)


def test_completed_batch_no_forward(tmp_path):
    selection=cached(tmp_path)
    vectors,_=module.visual_once(None,selection,tmp_path,'fine/q/0','0','fp')
    assert vectors.shape==(1,3)


def test_unknown_forward_not_repeated(tmp_path):
    selection=cached(tmp_path,'started')
    with pytest.raises(ValueError):module.visual_once(None,selection,tmp_path,'fine/q/0','0','fp')


def test_gpu_change_rejected(tmp_path):
    selection=cached(tmp_path)
    with pytest.raises(ValueError):module.visual_once(None,selection,tmp_path,'fine/q/0','1','fp')


def test_frame_identity_change_rejected(tmp_path):
    selection=cached(tmp_path);selection['selected_frames'][0]['source_pts']=2
    with pytest.raises(ValueError):module.visual_once(None,selection,tmp_path,'fine/q/0','0','fp')
