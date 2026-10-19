"""长视频试点：资产边界、乱序选项、标签隔离、门槛与一次训练恢复。"""
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from videoqa_set_value import longvideo_assets as assets
from videoqa_set_value import longvideo_trial as trial
from videoqa_set_value import longvideo_runtime as runtime
from videoqa_set_value import longvideo_task as task


def example():
    return dict(question_id='lvr-train-7',question='What changes?',options=['Fourth','First','Second','Third'],
        option_letters=list('DABC'),reasoning='DO NOT LEAK REASONING',answer_letter='B',reference_answer='DO NOT LEAK ANSWER')


def test_fixed_inventory():
    data=assets.inventory()
    assert len(data)==11
    assert sum(r['bytes'] for r in data[3:])==189521777735
    assert [r['name'].split('_')[-1].split('.')[0] for r in data[3:]]==list('nhZdqL0X')
    assert all(len(r['sha256'])==64 for r in data)
    assert all(assets.VIDEO_REVISION in r['url'] for r in data[3:])
    assert all(assets.ANNOTATION_REVISION in r['url'] for r in data[:3])


@pytest.mark.parametrize('suffix',['../secret','longvila_videos_a.tar.gz','config.json'])
def test_asset_whitelist(suffix):
    with pytest.raises(ValueError):assets.source_entry(suffix)


@pytest.mark.parametrize('newline',['\n','\\n','\r\n','\\r\\n'])
def test_permuted_options_and_escapes(newline):
    text=newline.join(['What changes?','D. Fourth','A. First','B. Second','C. Third'])
    value=trial.parse_problem(text)
    assert value=={k:example()[k] for k in ('question','options','option_letters')}


@pytest.mark.parametrize('text',[
    'Question\nA. One\nB. One\nC. Three\nD. Four',
    'Question\nA. One\nA. Two\nC. Three\nD. Four',
    'Question\nA. One\nB. Two\nC. Three',
    '\nA. One\nB. Two\nC. Three\nD. Four'])
def test_ambiguous_options_rejected(text):
    with pytest.raises(ValueError):trial.parse_problem(text)


def test_public_projection_and_original_letter():
    row=example();public=task.public_doc(row);text=task.context(row)
    assert set(public)=={'question','options','option_letters'}
    assert text.index('D. Fourth')<text.index('A. First')<text.index('B. Second')<text.index('C. Third')
    assert 'DO NOT LEAK' not in text
    assert task.metric(row,'B.')['correct']
    assert not task.metric(row,'A')['correct']
    assert not task.metric(row,'E')['correct']


@pytest.mark.parametrize('raw,pred',[('A','A'),(' C. ','C'),('(D)','D'),('E',None),
    ('D. explanation',None),('<answer>A</answer>',None),('',None)])
def test_strict_four_letter_parser(raw,pred):
    assert task.strict_option(raw)==pred


def test_label_is_loaded_only_at_scoring_boundary(tmp_path):
    row=example();row.pop('answer_letter');row['_scoring_root']=str(tmp_path)
    with patch.object(runtime,'scoring_row',side_effect=RuntimeError('scoring boundary')) as loader:
        task.context(row);task.public_doc(row);loader.assert_not_called()
        with pytest.raises(RuntimeError,match='scoring boundary'):task.metric(row,'A')


@pytest.mark.parametrize('value',['../abcdefghijk.mp4','longvila_videos/a.mp4','longvila_videos/abcdefghijk.webm','["a.mp4"]'])
def test_train_source_identity_is_strict(value):
    with pytest.raises(ValueError):trial.source_id(value)


def test_1200_hash_split_does_not_read_answers():
    groups=[];questions={}
    for i in range(1205):
        vid=str(i)
        groups.append(dict(video_id=vid,path=vid+'.mp4',sha256=str(i),index_path=vid+'.json',duration_seconds=200))
        questions[vid]=[dict(example(),video_id=vid,question_id='q'+vid,official_problem_id=i,reasoning='secret',answer='A')]
    a=trial.choose_split(groups,questions)
    for values in questions.values():values[0]['answer']='D';values[0]['reasoning']='changed'
    b=trial.choose_split(list(reversed(groups)),questions)
    assert a==b and len(a)==1200
    assert sum(r['pilot'] for r in a)==200
    assert sum(r['split']=='train' for r in a)==800
    assert len({r['content_group'] for r in a})==1200
    assert not any('reasoning' in r or 'answer' in r or 'answer_letter' in r for r in a)
    with pytest.raises(ValueError,match='fewer than 1200'):trial.choose_split(groups[:1199],questions)


def test_donor_cannot_use_validation():
    target=dict(example(),split='train',video_id='a',content_group='same',duration_seconds=200)
    same=dict(target,video_id='same-content',duration_seconds=201)
    val=dict(target,split='val',video_id='v',content_group='v',duration_seconds=200)
    train=dict(target,video_id='t',content_group='other',duration_seconds=300)
    assert runtime.donor_for(target,[target,same,val,train])==train


def test_chunk_receive_and_final_sha(tmp_path,monkeypatch):
    data=b'fixed asset bytes';sha=assets.hashlib.sha256(data).hexdigest()
    entry=dict(name='tiny.parquet',kind='annotation',bytes=len(data),sha256=sha,url='pinned')
    monkeypatch.setattr(assets,'source_entry',lambda _:entry)
    with pytest.raises(ValueError,match='Incomplete'):assets.receive_chunk(tmp_path,'tiny.parquet',0,len(data),sha,io.BytesIO(data[:-1]))
    assert not (tmp_path/'chunks/tiny.parquet/0.json').exists()
    with pytest.raises(ValueError,match='SHA'):assets.receive_chunk(tmp_path,'tiny.parquet',0,len(data),'f'*64,io.BytesIO(data))
    assets.receive_chunk(tmp_path,'tiny.parquet',0,len(data),sha,io.BytesIO(data))
    assert not assets.asset_state(tmp_path,'tiny.parquet')['complete']
    result=assets.finalize_asset(tmp_path,'tiny.parquet')
    assert Path(result['path']).read_bytes()==data
    assert assets.asset_state(tmp_path,'tiny.parquet')['complete']
    assert not (tmp_path/'chunks/tiny.parquet/0').exists()
    # 整包保留，恢复验证不产生下载或模型调用。
    assert assets.finalize_asset(tmp_path,'tiny.parquet')['sha256']==sha


@pytest.mark.parametrize('offset,length',[(1,5),(-1,4),(0,0),(0,1)])
def test_chunk_boundary_rejected(tmp_path,offset,length):
    with pytest.raises(ValueError):assets.receive_chunk(tmp_path,assets.inventory()[0]['name'],offset,length,'0'*64,io.BytesIO(b'x'))


def test_empty_or_mutated_review_cannot_pass(tmp_path):
    assert runtime.review_passes(tmp_path) is None
    panel=dict(question_ids=['q'+str(i) for i in range(12)])
    assets.atomic_json(tmp_path/'pilot_review_panel.json',panel)
    assets.atomic_json(tmp_path/'pilot_report.json',dict(mixed_six=panel['question_ids'][6:]))
    cases=[dict(question_id=q,supported=True,information_tradeoff=i>=6,evidence='visible frame evidence') for i,q in enumerate(panel['question_ids'])]
    review=dict(panel_sha256=assets.file_sha(tmp_path/'pilot_review_panel.json'),cases=cases,minutes=30)
    assets.atomic_json(tmp_path/'pilot_review.json',review)
    assert runtime.review_passes(tmp_path)['passed']
    review['cases'][0]['supported']=False;review['cases'][1]['supported']=False;review['cases'][2]['supported']=False
    assets.atomic_json(tmp_path/'pilot_review.json',review)
    assert not runtime.review_passes(tmp_path)['passed']
    review['minutes']=61;assets.atomic_json(tmp_path/'pilot_review.json',review)
    with pytest.raises(ValueError):runtime.review_passes(tmp_path)


def test_paired_interval_and_budgets():
    assert runtime.paired_interval(np.ones(200))==[1.,1.]
    assert sum(runtime.PHASE_CAPS.values())==5200
    assert runtime.PHASE_CAPS==dict(pilot=1200,train_rest=2400,val=1600)


def test_validation_protocol_requires_predictions(tmp_path):
    # 没有事前400题预测就无法发布验证阶段，不能仅靠训练结束绕过。
    with pytest.raises(FileNotFoundError):runtime.phase_protocol(tmp_path,'val')


def test_frozen_selection_modules_not_rewritten():
    path=Path(trial.__file__).with_name('longvideo_runtime.py')
    text=path.read_text()
    assert 'train_trajectory(data' in text
    assert 'rows(root,\'train\')' in text
    assert 'train_trajectory' not in Path(trial.__file__).with_name('longvideo_task.py').read_text()


def test_safe_tar_paths_and_fixed_member(tmp_path,monkeypatch):
    import tarfile
    path=tmp_path/'unsafe.tar.gz'
    with tarfile.open(path,'w:gz') as out:
        info=tarfile.TarInfo('../abcdefghijk.mp4');info.size=1;out.addfile(info,io.BytesIO(b'x'))
    entry=dict(kind='archive',sha256=assets.file_sha(path))
    monkeypatch.setattr(assets,'source_entry',lambda _:entry)
    monkeypatch.setattr(assets,'asset_state',lambda *_:dict(complete=True,path=str(path)))
    with pytest.raises(ValueError,match='Unsafe'):assets.extract_archive(tmp_path,'unsafe',{'abcdefghijk'})


def test_pilot_fixed_gates_and_review_order(tmp_path,monkeypatch):
    selected=[dict(question_id='q'+str(i),ordinal=i,pilot=True,split='train') for i in range(200)]
    y=np.ones((200,6),dtype=int);y[:,4:]=0;y[:20,1:4]=0;y[20:40,0]=0
    monkeypatch.setattr(runtime,'rows',lambda *args:selected)
    monkeypatch.setattr(runtime,'audit_phase',lambda *args:None)
    monkeypatch.setattr(runtime,'review_page',lambda *args:None)
    original=runtime.read_json
    def read(path):
        path=Path(path)
        if 'results' in path.parts:return dict(correct=bool(y[int(path.stem[1:]),runtime.CONDITIONS.index(path.parent.name)]))
        return original(path)
    monkeypatch.setattr(runtime,'read_json',read)
    # 真实调用预算与伪数据分离；输入哈希仅用于面板绑定。
    for row in selected:assets.atomic_json(tmp_path/'inputs/pilot'/(row['question_id']+'.json'),{})
    result=runtime.pilot_statistics(tmp_path)
    assert result['mixed_videos']==40 and result['gap_to_best_fixed']==20 and result['quantitative_pass']
    assert result['review_question_ids']==['q'+str(i) for i in range(12)]
    y[:,4]=y[:,0]
    assert not runtime.pilot_statistics(tmp_path)['quantitative_pass']


def test_raw_return_restores_without_new_generation(tmp_path,monkeypatch):
    row=dict(example(),question_id='q1');phase=tmp_path/'qa/pilot';method=runtime.METHODS[0]
    assets.atomic_json(phase/'protocol.json',{})
    fp=assets.file_sha(phase/'protocol.json');identity=dict(question_id='q1',method=method,gpu='0',protocol_sha256=fp)
    assets.atomic_json(phase/'calls'/method/'q1.json',dict(status='failed_or_uncertain',gpu='0',protocol_sha256=fp))
    assets.atomic_json(phase/'raw_generations'/method/'q1.json',dict(identity=identity,
        generation_state=dict(started=True,returned=True),raw_generated_text='B'))
    monkeypatch.setattr(task,'answer_from_raw',lambda raw,row:dict(raw_output=raw['raw_generated_text'],parsed_answer='B'))
    assert runtime.restore_raw(phase,row,method,'0')['parsed_answer']=='B'
    assert runtime.restore_raw(phase,row,method,'0')['parsed_answer']=='B'
    with pytest.raises(ValueError,match='GPU'):runtime.restore_raw(phase,row,method,'1')
    cp=phase/'calls'/method/'q1.json';value=json.loads(cp.read_text());value['status']='started_may_generate';assets.atomic_json(cp,value)
    (phase/'raw_generations'/method/'q1.json').unlink()
    with pytest.raises(RuntimeError,match='repetition forbidden'):runtime.restore_raw(phase,row,method,'0')


def test_validation_predictions_do_not_read_validation_feedback(tmp_path,monkeypatch):
    import torch
    from videoqa_set_value.learner import SetValueHead
    from videoqa_audit.set_value_trainability import TrainingAssets
    train=[dict(question_id='t',split='train',video_id='t')]
    x=np.zeros((1,4,8,770),dtype=np.float32);x[:,:,:,0]=1
    data=TrainingAssets(train,x,np.array([[1,0,1,0]],dtype=np.float32),[[0,1,2,3]],[[[[j]] for j in range(4)]])
    monkeypatch.setattr(runtime,'training_assets',lambda _:data)
    monkeypatch.setattr('videoqa_audit.set_value_trainability.train_trajectory',lambda *a,**k:dict(status='fit_passed'))
    selected=[dict(question_id='v',split='val',video_id='v')]
    monkeypatch.setattr(runtime,'rows',lambda *a:selected)
    (tmp_path/'head').mkdir();old=tmp_path/'old.pt'
    torch.save(dict(model=SetValueHead().state_dict()),tmp_path/'head/state.pt')
    torch.save(dict(model=SetValueHead().state_dict()),old)
    monkeypatch.setattr(runtime,'OLD_HEAD',old);monkeypatch.setattr(runtime,'OLD_HEAD_SHA',assets.file_sha(old))
    assets.atomic_json(tmp_path/'inputs/val/v.json',dict(methods={m:dict(source_order=[[j]]) for j,m in enumerate(runtime.METHODS)}))
    (tmp_path/'value_features').mkdir()
    np.savez(tmp_path/'value_features/v.npz',**{m:x[0,i] for i,m in enumerate(runtime.METHODS)})
    original=runtime.read_json;reads=[]
    def read(path):
        reads.append(str(path))
        if '/qa/val/' in str(path):raise AssertionError('Validation feedback accessed before predictions')
        return original(path)
    monkeypatch.setattr(runtime,'read_json',read)
    result=runtime.train_and_predict(tmp_path)
    assert result['status']=='ready_for_validation'
    value=json.loads((tmp_path/'validation_predictions.json').read_text())
    assert not value['validation_labels_read'] and len(value['predictions'])==1
    assert value['training_best_fixed_method']==runtime.METHODS[0]
    assert not any('/qa/val/' in p for p in reads)


def test_waiting_for_assets_does_not_allow_source_drift(tmp_path,monkeypatch):
    source=tmp_path/'repo';run=tmp_path/'run';old=tmp_path/'old.pt';old.write_bytes(b'frozen old head')
    names=['scripts/run_longvideo_set_value_trial.py','src/videoqa_audit/set_value_trainability.py',
        'src/videoqa_mlvu/data.py','src/videoqa_full/state.py','src/videoqa_full/prepare.py',
        'src/videoqa_lmms_bridge/common.py','src/videoqa_lmms_bridge/llava_adapter.py',
        'src/videoqa_generalization/llava_observe.py','src/videoqa_budget32/frames.py',
        'configs/retrieval_density_v1.json','configs/method_versions.json','configs/local_model_inventory.json',
        'configs/llava_source_manifest.json']
    for name in names:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixed source')
    monkeypatch.setattr(runtime,'ROOT',source);monkeypatch.setattr(runtime,'ENTRY',source/names[0])
    monkeypatch.setattr(runtime,'OLD_HEAD',old);monkeypatch.setattr(runtime,'OLD_HEAD_SHA',assets.file_sha(old))
    monkeypatch.setattr(runtime,'framework_identity',lambda:dict(commit='fixed'))
    first=runtime.freeze_sources(run)
    assert runtime.freeze_sources(run)==first and first['new_model_calls']==0
    (source/names[0]).write_text('changed source')
    with pytest.raises(ValueError,match='Pre-download frozen source changed'):runtime.freeze_sources(run)


def test_known_bank_includes_unselected_and_bad_old_assets(tmp_path,monkeypatch):
    ref=tmp_path/'r1/reference_bank.json';old=tmp_path/'r2';repo=tmp_path/'repo';fp=dict(phash=['0'*16]*8)
    assets.atomic_json(ref,dict(videos=[dict(source_id='target:held',video_sha256='held-sha',fingerprint=fp)]))
    assets.atomic_json(old/'manifest.json',dict(rows=[dict(video_id='1',video_sha256='sha1')]))
    for vid,eligible in [('1',True),('2',False),('3',True)]:
        value=dict(video_id=vid,video_sha256='sha'+vid,eligible=eligible)
        if eligible:value['fingerprint']=fp
        assets.atomic_json(ref.parent/'candidate_checks'/(vid+'.json'),value)
    assets.atomic_json(repo/'data/manifests/video_mme_full_2700.json',dict(rows=[dict(video_id='yt-old')]))
    monkeypatch.setattr(trial,'REFERENCE',ref);monkeypatch.setattr(trial,'OLD_DATA',old);monkeypatch.setattr(trial,'ROOT',repo)
    bank,ids,sources,hashes=trial.known_content()
    assert len(bank)==3 and {'1','2','3','held','yt-old'}<=ids
    assert {'held-sha','sha1','sha2','sha3'}==hashes
    assert all('fingerprint' in r for r in bank)
