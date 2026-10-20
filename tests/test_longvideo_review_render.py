"""固定面板渲染验收：来源篡改拒绝，不引入解码、模型或审核通过标记。"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


def load():
    """测试接口：导入独立显示工具，不运行真实试题审核。"""
    path=Path(__file__).resolve().parents[1]/'scripts/render_longvideo_pilot_review.py'
    spec=importlib.util.spec_from_file_location('longvideo_review_renderer',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture_bank(tmp_path):
    """测试接口：仅有八张合成小图，元数据和像素摘要都必须匹配。"""
    frames=[];files={}
    for i in range(8):
        p=tmp_path/(str(i)+'.png');image=Image.new('RGB',(24,16),(i*20,10,30));image.save(p)
        rgb=hashlib.sha256(np.asarray(image).tobytes()).hexdigest();image.close()
        frames.append(dict(image=str(p),rgb_sha256=rgb,source_pts=i,source_frame_index=i,timestamp_seconds=float(i)))
        files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(frames=frames),dict(files=files)


def test_board_keeps_frozen_sources_and_no_gate(tmp_path):
    m=load();record,checkpoint=fixture_bank(tmp_path);before=[Path(f['image']).read_bytes() for f in record['frames']]
    result=m.draw_bank(record,checkpoint,tmp_path,tmp_path/'review/board.jpg')
    assert result['source_pts']==list(range(8)) and Path(result['path']).is_file()
    assert before==[Path(f['image']).read_bytes() for f in record['frames']]
    assert not (tmp_path/'pilot_review.json').exists()


def test_changed_png_or_rgb_rejected(tmp_path):
    m=load();record,checkpoint=fixture_bank(tmp_path);record['frames'][0]['rgb_sha256']='wrong'
    with pytest.raises(ValueError,match='RGB'):m.draw_bank(record,checkpoint,tmp_path,tmp_path/'bad.jpg')
    record,checkpoint=fixture_bank(tmp_path);Path(record['frames'][0]['image']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='image changed'):m.draw_bank(record,checkpoint,tmp_path,tmp_path/'bad.jpg')


def test_missing_fixed_panel_cannot_render(tmp_path):
    with pytest.raises(FileNotFoundError):load().render(tmp_path)
