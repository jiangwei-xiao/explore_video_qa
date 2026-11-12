"""独立后台审计清单隔离与加载边界。"""
import importlib.util
from pathlib import Path


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/run_selection_structure_independent_audit.py'
    spec=importlib.util.spec_from_file_location('independent_audit_coordinator_test',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_scientific_inventory_excludes_live_status_and_its_own_manifest(tmp_path):
    files=['qa/llava8/results/m/q.json','qa/qwen32/raw_generations/m/q.json','code_snapshot/src/a.py',
           'offline_protocol.json','summary.json','independent_audit/llava8.json',
           'pipeline_status.json','independent_audit/restore_before.json','qa/llava8/logs/gpu0.log']
    for name in files:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    got={str(p.relative_to(tmp_path)) for p in module().scientific_files(tmp_path)}
    assert got==set(files[:6])
