"""完整粗池对照的来源复用测试：不伪造新调用、不接受变化的源返回。"""
import importlib.util
from pathlib import Path
import pytest
from videoqa_full.state import durable
from videoqa_runtime.common import sha256

spec=importlib.util.spec_from_file_location('event_full_probe',Path(__file__).resolve().parents[1]/'scripts/probe_event_clause_full_coarse.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def imported(tmp_path):
    path=tmp_path/'parent.json';feature=tmp_path/'parent.npy'
    durable(path,dict(status='completed',scores=[0.2,0.3]));feature.write_bytes(b'known feature bytes')
    item=dict(path=str(path),sha256=sha256(path),feature_path=str(feature),feature_sha256=sha256(feature))
    return dict(imports={'q':{'event_clause':{'0':item}}}),path,feature


def test_completed_import_does_not_create_new_ledger(tmp_path):
    spec,path,_=imported(tmp_path)
    result=m.returned(tmp_path/'new','q','event_clause',0,spec)
    assert result['scores']==[0.2,0.3]
    assert not (tmp_path/'new/calls').exists()


@pytest.mark.parametrize('which',['return','feature'])
def test_changed_import_pauses_not_rescores(tmp_path,which):
    spec,path,feature=imported(tmp_path)
    if which=='return':durable(path,dict(status='completed',scores=[0.8]))
    else:feature.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Imported score changed'):
        m.returned(tmp_path/'new','q','event_clause',0,spec)
