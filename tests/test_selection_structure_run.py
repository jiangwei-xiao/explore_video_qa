"""来源闭合、分组、冻结与恢复边界的CPU验收，不以合成结果冒充远端全量。"""
import copy
import importlib.util
import json
from pathlib import Path
from concurrent.futures import Future

import numpy as np
import pytest

from videoqa_audit import selection_structure as method
from videoqa_audit import selection_structure_run as study
from videoqa_audit import selection_structure_runtime as runtime
from videoqa_audit import selection_structure_report as report


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def physical_fixture(root,method_id='RD-1.2-exp20',representative='A'):
    """接口：构造最小物理账本/原始返回契约，用于校验引用链边界。"""
    q='001-1';spec=put(root/'protocol.json',dict(methods={'A':method_id}))
    identity=dict(question_id=q,method=method_id,protocol_sha256=study.sha(spec),gpu='2')
    answer=dict(raw_output='C',observed=dict(input_token_ids=[2,3]))
    value=dict(question_id=q,method=method_id,representative=representative,protocol_sha256=study.sha(spec),
               gpu='2',answer=answer,reference_sources={})
    path=put(root/'results'/method_id/(q+'.json'),value)
    cp=put(root/'calls'/method_id/(q+'.json'),dict(identity,status='completed',answer=answer))
    rp=put(root/'raw_generations'/method_id/(q+'.json'),dict(identity=identity,generation_state=dict(started=True,returned=True)))
    return path,value,cp,rp


def test_atomic_freeze_is_read_only_on_resume(tmp_path):
    path=tmp_path/'frozen.json';value=dict(question_id='001-1',frame_budget=8)
    study.freeze(path,value); before=(path.read_bytes(),path.stat().st_mtime_ns)
    study.freeze(path,dict(reversed(list(value.items()))))
    assert before==(path.read_bytes(),path.stat().st_mtime_ns)
    with pytest.raises(ValueError): study.freeze(path,dict(value,frame_budget=32))
    assert before==(path.read_bytes(),path.stat().st_mtime_ns)
    with pytest.raises(ValueError): study.freeze(tmp_path/'bad.json',dict(value=float('nan')))
    assert not (tmp_path/'bad.json').exists()


def test_coarse_mapping_is_after_pool_mapping():
    pool=dict(initial_count=2,candidates=[{'candidate_index':i} for i in range(3)])
    original=np.array([[1,2],[3,4],[5,6],[7,8]],dtype=np.float32)
    mapped=original[[2,0,3]];coarse,features=study.coarse_view(pool,mapped)
    np.testing.assert_array_equal(features,[[5,6],[1,2]])
    assert len(coarse['candidates'])==2 and len(pool['candidates'])==3
    with pytest.raises(ValueError): study.coarse_view(pool,original)


def test_trace_discrete_zero_tolerance():
    old=dict(region_budgets={'0':2},curve=[0.,.5],prefix=[1,3],control='A',selection_seconds=4.)
    new=dict(region_budgets={'0':2},curve=[0.,.5000005],prefix=[1,3])
    study.compare_trace(new,old)
    with pytest.raises(ValueError,match='discrete'): study.compare_trace(dict(new,prefix=[1,2]),old)
    with pytest.raises(ValueError,match='numeric'): study.compare_trace(dict(new,curve=[0.,.500002]),old)
    with pytest.raises(ValueError,match='Missing'): study.compare_trace({},old)


def test_grouping_has_fixed_representative_and_no_label_dependency():
    items={m:dict(identity=dict(pixels='same',tokens=[7]),correct=(m==method.WFS)) for m in method.CONTROLS}
    groups=runtime.input_groups(items)
    assert len(groups)==1 and groups[0]['representative']==method.GLOBAL
    changed=copy.deepcopy(items)
    for v in changed.values(): v['correct']=not v['correct']
    assert runtime.input_groups(changed)==groups
    changed[method.WFS]['identity']['pixels']='different'
    assert len(runtime.input_groups(changed))==2


def test_call_population_rejects_extra_or_duplicated_slots(tmp_path):
    spec=dict(new_groups={'001-1':[method.GLOBAL]},max_new_calls=1)
    assert runtime.validate_call_population(tmp_path,spec)=={(method.GLOBAL,'001-1')}
    with pytest.raises(ValueError): runtime.validate_call_population(tmp_path,spec,complete=True)
    put(tmp_path/'calls'/method.GLOBAL/'001-1.json',{})
    put(tmp_path/'raw_generations'/method.GLOBAL/'001-1.json',{})
    runtime.validate_call_population(tmp_path,spec,complete=True)
    put(tmp_path/'calls'/method.GLOBAL/'002-1.json',{})
    with pytest.raises(ValueError): runtime.validate_call_population(tmp_path,spec)
    with pytest.raises(ValueError):
        runtime.validate_call_population(tmp_path,dict(new_groups={'001-1':[method.GLOBAL,method.GLOBAL]},max_new_calls=2))


def test_physical_ledger_maps_old_control_alias(tmp_path):
    path,value,cp,rp=physical_fixture(tmp_path)
    result=runtime.physical_sources(path,value)
    assert result[str(cp)]==study.sha(cp) and result[str(rp)]==study.sha(rp)
    assert str(tmp_path/'protocol.json') in result


@pytest.mark.parametrize('damage',['status','raw_state','gpu','answer','protocol'])
def test_uncertain_or_mismatched_physical_return_rejected(tmp_path,damage):
    path,value,cp,rp=physical_fixture(tmp_path)
    c=study.read(cp);raw=study.read(rp)
    if damage=='status': c['status']='failed_or_uncertain';put(cp,c)
    if damage=='raw_state': raw['generation_state']['returned']=False;put(rp,raw)
    if damage=='gpu': raw['identity']['gpu']='7';put(rp,raw)
    if damage=='answer': c['answer']['raw_output']='A';put(cp,c)
    if damage=='protocol': value['protocol_sha256']='wrong'
    with pytest.raises(ValueError): runtime.physical_sources(path,value)


def test_import_chain_requires_actual_matching_return(tmp_path):
    path,value,cp,rp=physical_fixture(tmp_path/'parent')
    bundle=runtime.physical_sources(path,value)
    newroot=tmp_path/'new';spec=put(newroot/'protocol.json',dict(methods=list(method.CONTROLS)))
    imported=dict(value,method=method.GLOBAL,representative=method.GLOBAL,
        protocol_sha256=study.sha(spec),reference_sources=bundle)
    target=put(newroot/'results'/method.GLOBAL/'001-1.json',imported)
    assert str(cp) in runtime.physical_sources(target,imported)
    raw=study.read(rp);raw['generation_state']['returned']=False;put(rp,raw)
    with pytest.raises(ValueError): runtime.physical_sources(target,imported)


def test_byte_copied_import_uses_explicit_origin_protocol(tmp_path):
    origin=tmp_path/'original';path,value,cp,rp=physical_fixture(origin,representative=method.PARENT)
    batch=tmp_path/'batch';files={}
    for source in (path,cp,rp):
        dest=batch/'imported'/source.relative_to(origin)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        files[str(dest.relative_to(batch))]=study.sha(dest)
    put(batch/'protocol.json',dict(imports={'001-1':dict(origin=str(origin),files=files,
                                                      protocol_sha256=value['protocol_sha256'])}))
    copied=batch/'imported'/path.relative_to(origin)
    bundle=runtime.physical_sources(copied,value)
    assert str(origin/'protocol.json') in bundle and str(batch/'protocol.json') in bundle
    changed=study.read(batch/'protocol.json');changed['imports']['001-1']['origin']=str(tmp_path/'unrelated')
    put(batch/'protocol.json',changed)
    with pytest.raises((ValueError,FileNotFoundError)): runtime.physical_sources(copied,value)


def test_posthoc_mapping_does_not_impose_parent_prefix():
    trace=dict(region_budgets={'0':2,'1':1},protected_indices=[0,3],coarse_prefix_trace=dict(regions=[
        dict(region_id=0,reference_coarse_indices=[0,1,2],prefix=[0,2]),
        dict(region_id=1,reference_coarse_indices=[3,4],prefix=[3,4])]))
    mapped=study.map_to_parent([1,3,4],trace,[0,3]);a,b=mapped['regions']
    assert a['selected_prefix_positions']=={'1':None}
    assert b['control_count']==2 and b['parent_quota']==1
    assert mapped['anchor_retained']==[3]


def test_completed_offline_verification_never_calls_selector(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    code=put(tmp_path/'code.json',{'file':'body'});source=put(tmp_path/'source.json',{'pool':'identity'})
    target=put(tmp_path/'selections/q.json',{'selected':[1,3]})
    study.snapshot_code(tmp_path,{'code.json':study.sha(code)})
    put(tmp_path/'offline_protocol.json',dict(code_sha256={'code.json':study.sha(code)}))
    put(tmp_path/'offline_summary.json',dict(parent_replays=5400,control_selections=10800,
        protocol_sha256=study.sha(tmp_path/'offline_protocol.json'),sources={str(source):study.sha(source)},
        exports={'selections/q.json':study.sha(target)}))
    def forbidden(*args,**kwargs): raise AssertionError('Selector must not run during completed verification')
    monkeypatch.setattr(method,'select_coarse_control',forbidden)
    files=[p for p in tmp_path.rglob('*') if p.is_file()]
    before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in files}
    study.verify_offline(tmp_path)
    assert before=={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in files}
    put(target,{'selected':[2,4]})
    with pytest.raises(ValueError,match='export'): study.verify_offline(tmp_path)


def test_fixed_pipeline_stages_and_resume_flags():
    path=Path(__file__).resolve().parents[1]/'scripts/run_selection_structure_diagnostic.py'
    spec=importlib.util.spec_from_file_location('selection_structure_entry_test',path)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    commands=entry.commands('fixed_run',['0','3'])
    names=[name for name,_ in commands]
    assert names.index('freeze_calls')<names.index('answer_llava8')<names.index('answer_qwen32')
    assert names.index('prepare_qwen32')<names.index('freeze_calls')
    assert all('--resume' in cmd for _,cmd in commands)
    assert dict(commands)['panel'][0]==study.PYTHONS['llava8']
    assert dict(commands)['report'][0]==study.PYTHONS['qwen32']


def test_cpu_failure_does_not_enqueue_entire_matrix():
    class ImmediateExecutor:
        def __init__(self): self.started=[]
        def submit(self,fn,job):
            self.started.append(job);f=Future()
            try: f.set_result(fn(job))
            except Exception as exc: f.set_exception(exc)
            return f
    executor=ImmediateExecutor()
    def reject_first(job):
        if job==0: raise ValueError('source mismatch')
        return job
    with pytest.raises(ValueError,match='source mismatch'):
        list(study.bounded_results(executor,reject_first,range(5400),4))
    assert executor.started==[0,1,2,3]


def test_panel_avoids_duplicate_video_and_ambiguous_questions():
    data=[dict(question_id=q,video_id=v) for q,v in [('491-2','491'),('001-1','1'),('001-2','1'),('002-1','2')]]
    comparisons={p:{method.PARENT+'__minus__'+c:dict(cells=dict(left_only=['491-2','001-1','001-2'],right_only=['002-1']))
                   for c in method.CONTROLS} for p in study.PROFILES}
    entries=report.panel_entries(data,comparisons)
    assert len(entries)==2 and len({e['video_id'] for e in entries})==2
    assert '491-2' not in {e['question_id'] for e in entries}
