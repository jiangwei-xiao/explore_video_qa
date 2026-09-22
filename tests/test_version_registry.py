"""版本登记一致性、别名歧义与报告路径验收。"""
from pathlib import Path
import pytest
from videoqa_audit.version_registry import load_registry, canonical_name, REGISTRY

@pytest.mark.parametrize('alias,scope,expected',[
    ('A','sb','SB-1.0'),('B','sb','SB-1.1'),('C','sb','SB-1.2'),('E','sb','SB-1.3'),
    ('D','sb','SB-1.2-diag01'),('Q','sb','SB-1.3-exp01'),('C-local',None,'SB-1.3'),
    ('v1','rd','RD-1.0'),('soft','rd','RD-1.1'),('P1','rd','RD-1.1-exp01'),('D1','rd','RD-1.1-exp02'),
    ('E1','rd_offline','RD-1.1-exp03'),('E1','rd_formal','RD-1.2'),('RD-2P v1.0',None,'RD-1.2'),
    ('TopK',None,'BASE-BLIP-TopK'),('uniform',None,'BASE-Uniform'),('R','rb','RB-1.0')])
def test_scoped_aliases(alias,scope,expected):
    assert canonical_name(alias,scope)==expected

@pytest.mark.parametrize('value',['A','B','E','D','Q','R','v1','E1','unregistered'])
def test_never_guess_ambiguous_tokens(value):
    with pytest.raises(ValueError):canonical_name(value)

def test_registry_references_and_idempotence():
    data=load_registry();entries=data['entries'];ids={e['id'] for e in entries};root=REGISTRY.parent.parent
    assert len(ids)==len(entries) and data['current_frozen_method']=='RD-1.2'
    for entry in entries:
        assert canonical_name(entry['id'])==entry['id']
        assert entry['parent'] is None or entry['parent'] in ids
        assert (root/entry['report']).is_file()
        for ref in entry.get('dependencies',[])+entry.get('adopted_explorations',[]):assert ref in ids
        if 'selection_base' in entry:assert entry['selection_base'] in ids
    byid={e['id']:e for e in entries}
    assert byid['SB-1.3-exp01']['selection_base']=='SB-1.1'
    assert byid['RD-1.1-exp03']['dependencies']==['RD-1.1-exp02']
    assert byid['RD-1.2']['legacy_registration']=='configs/releases/rd2p_v1_0.json'

def test_aliases_unique_inside_scope():
    seen={}
    for entry in load_registry()['entries']:
        for alias in entry['aliases']:
            key=(alias['scope'],alias['label'].casefold())
            assert key not in seen or seen[key]==entry['id']
            seen[key]=entry['id']
