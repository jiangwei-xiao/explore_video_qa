"""exp33小测运行边界：引用优先级、原文恢复、任务隔离和固定调用规模，零模型。"""
import copy
from pathlib import Path
from unittest.mock import Mock
import pytest
from videoqa_audit import adjacent_context_development as dev
from videoqa_full.state import durable


@pytest.mark.parametrize('cell',dev.CELLS)
def test_fixed_partition_closes_before_generation(cell):
    n,parent,diag,new=dev.EXPECTED[cell]
    items=[dict(origin=k) for k,count in [('parent_import',parent),('diagnostic_import',diag),('new_generation',new)] for _ in range(count)]
    assert dev.check_partition(*cell,items)['new_calls']==new
    items[-1]['origin']='parent_import'
    with pytest.raises(ValueError):dev.check_partition(*cell,items)


def selection(points):
    """接口：只用于隔离测试的源时钟与身份，不读取任何真实题或视频。"""
    return dict(video=dict(path='synthetic.mp4',start_pts=0,time_base='1/10',duration_fraction='10'),
                selected_frames=[dict(source_pts=p,source_frame_index=p,timestamp_seconds=p/10) for p in points])


def test_media_key_ignores_candidate_number_but_not_order():
    a=selection([1,2]);b=copy.deepcopy(a);b['selected_frames'][0]['candidate_index']=200
    assert dev.media_key(a)==dev.media_key(b)
    b['selected_frames'].reverse();assert dev.media_key(a)!=dev.media_key(b)


def test_dataset_model_namespaces_and_frozen_assignment():
    assert len({str(dev.unit_path(Path('/tmp/run'),*c)) for c in dev.CELLS})==4
    spec=dict(question_ids=['q'+str(i) for i in range(8)],new_ids=['q0','q2','q4','q5','q6','q7'])
    items=[dict(question_id=q,origin='new_generation' if q in spec['new_ids'] else 'parent_import',
                selection=dict(video=dict(duration_seconds=10+i))) for i,q in enumerate(spec['question_ids'])]
    plans=dev.assign_work(spec,items,['0','1']);allids=[q for p in plans.values() for phase in ('pilot','body') for q in p[phase]]
    assert len(allids)==len(set(allids))==6 and set(allids)==set(spec['new_ids'])
    assert {q for p in plans.values() for q in p['pilot']}=={'q0','q2','q4'}
    for p in plans.values():assert p['body']==sorted(p['body'],reverse=True)


@pytest.mark.parametrize('origin',['parent_import','diagnostic_import','new_generation'])
def test_prepare_uses_parent_first_then_exact_diagnostic(tmp_path,monkeypatch,origin):
    root=tmp_path/'run';source=tmp_path/'frozen';diagroot=tmp_path/'diag';profile='llava8';dataset='videomme';q='q1'
    monkeypatch.setattr(dev,'SOURCE',source);monkeypatch.setattr(dev,'DIAGNOSTIC',diagroot)
    new=selection([1,3]);old=copy.deepcopy(new) if origin=='parent_import' else selection([1,2]);ident=dict(prompt='q',input_token_ids=[4],pixel_sha256='actual')
    parentpath=tmp_path/'parent.json';answer=dict(raw_output='A',parsed_answer='A',observed={})
    durable(parentpath,dict(selection=old,rgb_sha256=['same','same' if origin=='parent_import' else 'old'],answer=answer,gpu='0'))
    itempath=source/'offline'/dataset/profile/(q+'.json')
    durable(itempath,dict(selection=new,parent_result=str(parentpath),trace=dict(changed=origin!='parent_import',changed_slots=[] if origin=='parent_import' else [1]),qwen_pairs_before=[],qwen_pairs_after=[]))
    durable(source/'offline_summary.json',dict(exports={'offline/'+dataset+'/'+profile+'/'+q+'.json':dev.sha256(itempath)}))
    video=tmp_path/'video.json';durable(video,{})
    monkeypatch.setattr(dev,'video_identity',lambda *a:video)
    class Image:
        def close(self):pass
    class Provider:
        def __init__(self,s):pass
        def decode(self):return [Image(),Image()],['same','same']
    monkeypatch.setattr(dev,'FrameProvider',Provider);monkeypatch.setattr(dev,'validate_frames',lambda *a:None)
    monkeypatch.setattr(dev,'cpu_input',lambda *a:ident);monkeypatch.setattr(dev,'observed_input',lambda *a:ident)
    monkeypatch.setattr(dev,'history_sources',lambda *a:{})
    import videoqa_audit.frame_group_diagnostic as diagnostic
    case=dict(id='D3',dataset=dataset,profile=profile,question_id=q,gpu='1')
    monkeypatch.setattr(diagnostic,'cases',lambda:[case])
    durable(diagroot/'D3/inputs/T.json',dict(selection=new if origin=='diagnostic_import' else selection([2,4]),rgb_sha256=['same','same'],input_identity=ident))
    model=dict(model_files={},framework={},environment={});durable(root/'model_identity/llava8.json',model);durable(diagroot/'D3/protocol.json',model)
    durable(diagroot/'D3/results/T.json',dict(answer=answer));durable(diagroot/'D3/calls/T/q1.json',dict(status='completed',answer=answer))
    durable(diagroot/'D3/raw_generations/T/q1.json',dict(generation_state=dict(started=True,returned=True)))
    result=dev.prepare_question((str(root),dataset,profile,dict(question_id=q)))
    assert result['origin']==origin
    assert (result['import_answer'] is not None)==(origin!='new_generation')


def test_complete_raw_return_recovers_postprocessing_only(tmp_path,monkeypatch):
    run=tmp_path/'cell';q='q1';spec=dict(dataset='videomme',profile='llava8');durable(run/'protocol.json',spec)
    durable(run/'inputs/q1.json',dict(row=dict(question_id=q)))
    fp=dev.sha256(run/'protocol.json');call=dict(protocol_sha256=fp,status='failed_or_uncertain',gpu='2')
    durable(run/'calls'/dev.METHOD/(q+'.json'),call)
    durable(run/'raw_generations'/dev.METHOD/(q+'.json'),dict(identity=dict(question_id=q,method=dev.METHOD,protocol_sha256=fp,gpu='2'),generation_state=dict(started=True,returned=True)))
    restore=Mock(return_value=dict(raw_output='A'));persist=Mock();monkeypatch.setattr(dev,'restore_answer',restore);monkeypatch.setattr(dev,'persist_result',persist)
    dev.recover_call(run,q)
    assert restore.call_count==1 and persist.call_count==1
    assert dev.read_json(run/'calls'/dev.METHOD/(q+'.json'))['status']=='completed'


def test_unknown_raw_return_never_reissues(tmp_path,monkeypatch):
    run=tmp_path/'cell';q='q1';durable(run/'protocol.json',dict(dataset='videomme',profile='llava8'));durable(run/'inputs/q1.json',dict(row={}))
    durable(run/'calls'/dev.METHOD/(q+'.json'),dict(protocol_sha256=dev.sha256(run/'protocol.json'),status='started_may_generate',gpu='0'))
    restore=Mock();monkeypatch.setattr(dev,'restore_answer',restore)
    with pytest.raises(RuntimeError,match='no retry'):dev.recover_call(run,q)
    assert not restore.called


@pytest.mark.parametrize('encoded,valid',[(12,True),(11,False)])
def test_qwen_odd_frame_padding_uses_actual_count(monkeypatch,encoded,valid):
    ident=dict(prompt='q');answer=dict(raw_output='A',parsed_answer='A',framework_metric=dict(pred_answer='A'),observed=dict(source_frames=11,generate_calls=1,prefill_calls=1,generated_token_ids=[5],generation_kwargs=dict(max_new_tokens=16,do_sample=False,num_beams=1),encoding_fps=2,encoded_frames=encoded,internal_padding_frames=1))
    item=dict(input_identity=ident,selection=selection(list(range(11))),row={})
    monkeypatch.setattr(dev,'observed_input',lambda *a:ident)
    import videoqa_mlvu.backends as ml
    monkeypatch.setattr(ml,'metric',lambda *a:dict(pred_answer='A'))
    if valid:dev.validate_answer(item,answer,'mlvu','qwen32')
    else:
        with pytest.raises(ValueError,match='padding'):dev.validate_answer(item,answer,'mlvu','qwen32')


def test_shared_cache_waits_for_existing_writer(tmp_path):
    import threading
    entered=threading.Event();started=threading.Event()
    def request():
        started.set()
        with dev.cache_lock(tmp_path/'same.lock'):entered.set()
    with dev.cache_lock(tmp_path/'same.lock'):
        thread=threading.Thread(target=request);thread.start();assert started.wait(2)
        assert not entered.wait(.1)
    assert entered.wait(2);thread.join(2);assert not thread.is_alive()


def test_no_idle_gpu_is_wait_not_coordinator_failure(monkeypatch):
    import videoqa_runtime.baseline_run as base
    def unavailable(gpus):raise RuntimeError('No requested GPU is currently idle')
    monkeypatch.setattr(base,'available_gpus',unavailable)
    assert dev.idle_worker_gpus()==[]


def test_gpu_query_errors_are_not_swallowed(monkeypatch):
    import videoqa_runtime.baseline_run as base
    def broken(gpus):raise RuntimeError('Driver unavailable')
    monkeypatch.setattr(base,'available_gpus',broken)
    with pytest.raises(RuntimeError,match='Driver unavailable'):dev.idle_worker_gpus()
