"""验收后的重建入口必须拒绝覆盖已封存交付，不加载任何模型。"""
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_assemble_refuses_frozen_output(tmp_path):
    """封存标记存在时，在读历史实验或写组表之前停止。"""
    out=tmp_path/'final_review';out.mkdir();(out/'acceptance.json').write_text('{}')
    rules=tmp_path/'rules.json';rules.write_text('{}')
    fn=runpy.run_path(str(ROOT/'scripts/assemble_final_audit.py'))['assemble']
    with pytest.raises(RuntimeError,match='封存'):fn(tmp_path,tmp_path/'missing',rules)
    assert sorted(p.name for p in out.iterdir())==['acceptance.json']


def test_renderer_refuses_frozen_output(tmp_path):
    """拒绝覆盖已有最终页面，连新子目录也不创建。"""
    out=tmp_path/'final_review';out.mkdir();(out/'acceptance.json').write_text('{}')
    fn=runpy.run_path(str(ROOT/'scripts/render_final_audit.py'))['render']
    with pytest.raises(RuntimeError,match='封存'):fn(tmp_path)
    assert sorted(p.name for p in out.iterdir())==['acceptance.json']
