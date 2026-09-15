"""审计模块测试：隔离合成数据，不调用真实问答。"""
import copy
import json
from pathlib import Path
import pytest
from videoqa_audit.core import validate_annotation, selected, write, read, digest, import_review


@pytest.fixture
def sample():
    """构造两帧的最小标注约束样本，仅用于验证器。"""
    return dict(question_id='q',frames=[dict(source_pts=1),dict(source_pts=2)],groups={'G1':[1,2]})


def test_pending_is_not_uncertain(sample):
    validate_annotation(sample,dict(question_id='q',status='pending',frames={},groups={}))


@pytest.mark.parametrize('label',['wrong','',None])
def test_invalid_label(sample,label):
    with pytest.raises(ValueError):validate_annotation(sample,dict(question_id='q',frames={'1':dict(label=label,reason='说明')}))


def test_unknown_pts(sample):
    with pytest.raises(ValueError):validate_annotation(sample,dict(question_id='q',frames={'3':dict(label='direct',reason='说明')}))


def test_incomplete_cannot_be_complete(sample):
    with pytest.raises(ValueError):validate_annotation(sample,dict(question_id='q',status='initial_reviewed',frames={},groups={}))


def test_context_override_is_group_local(sample):
    a=dict(question_id='q',frames={'1':dict(label='direct',reason='物体')},groups={'G1':dict(sufficiency='partial',reason='缺过程',overrides={'1':dict(label='context',reason='本组仅定位')})})
    validate_annotation(sample,a)


def test_redundancy_required(sample):
    a=dict(question_id='q',status='initial_reviewed',frames={str(p):dict(label='direct',reason='形状') for p in (1,2)},groups={'G1':dict(sufficiency='sufficient',reason='清楚')})
    with pytest.raises(ValueError):validate_annotation(sample,a)


def test_baseline_and_method_shape():
    assert selected({'selection':{'selected_frames':[1]}})==[1]
    assert selected({'pool':{'selected_frames':[2]}})==[2]


def test_atomic_json(tmp_path):
    write(tmp_path/'a.json',{'中文':'内容'});assert read(tmp_path/'a.json')=={'中文':'内容'}
    assert not (tmp_path/'a.json.tmp').exists()


def test_review_wrong_protocol_rejected(tmp_path):
    write(tmp_path/'protocol.json',{'v':1});write(tmp_path/'input.json',{'protocol_sha256':'wrong'})
    with pytest.raises(ValueError):import_review(tmp_path,tmp_path/'input.json')


def test_module_has_no_model_backend_import():
    import videoqa_audit.core as core
    source=Path(core.__file__).read_text()
    assert 'llava_backend' not in source and 'BlipFor' not in source


def test_import_is_idempotent_and_preserves_initial(tmp_path,sample):
    write(tmp_path/'protocol.json',{'v':1});write(tmp_path/'questions.json',[sample])
    initial=dict(question_id='q',status='pending',frames={'1':dict(label='direct',reason='可见')},groups={})
    write(tmp_path/'annotations/q.json',initial)
    review=dict(question_id='q',protocol_sha256=digest(tmp_path/'protocol.json'),initial_annotation_sha256=digest(tmp_path/'annotations/q.json'),groups={'G1':dict(sufficiency='',frames={'1':dict(label='direct'),'2':dict(label='')})})
    write(tmp_path/'input.json',review);import_review(tmp_path,tmp_path/'input.json');import_review(tmp_path,tmp_path/'input.json')
    assert read(tmp_path/'annotations/q.json')==initial
    assert len(list((tmp_path/'reviews').glob('*.agreement.json')))==1
    agreement=read(next((tmp_path/'reviews').glob('*.agreement.json')))
    assert agreement['compared_slots']==1 and agreement['agree_slots']==1


def test_review_rejects_foreign_frame(tmp_path,sample):
    write(tmp_path/'protocol.json',{'v':1});write(tmp_path/'questions.json',[sample]);write(tmp_path/'annotations/q.json',{})
    write(tmp_path/'input.json',dict(question_id='q',protocol_sha256=digest(tmp_path/'protocol.json'),groups={'G1':dict(sufficiency='',frames={'3':dict(label='direct')})}))
    with pytest.raises(ValueError):import_review(tmp_path,tmp_path/'input.json')


def test_manual_index_overlap_rejected():
    from videoqa_audit.observations import indices
    assert indices('1-3,5')==[1,2,3,5]
    with pytest.raises(ValueError):indices('1-3,3')


@pytest.mark.parametrize('version',['historical','missing','legacy'])
def test_agreement_uses_reviewed_version(tmp_path,sample,version):
    """初审修订后仍比较原版本；缺版本时保留复核而不制造一致率。"""
    write(tmp_path/'protocol.json',{'v':1});write(tmp_path/'questions.json',[sample])
    initial=dict(question_id='q',frames={'1':dict(label='direct',reason='可见')},groups={})
    write(tmp_path/'annotations/q.json',initial);old_hash=digest(tmp_path/'annotations/q.json')
    if version=='historical':write(tmp_path/f'annotation_history/q/{old_hash}.json',initial)
    write(tmp_path/'annotations/q.json',dict(question_id='q',frames={'1':dict(label='irrelevant',reason='已修订')},groups={}))
    review=dict(question_id='q',protocol_sha256=digest(tmp_path/'protocol.json'),groups={'G1':dict(sufficiency='',frames={'1':dict(label='direct'),'2':dict(label='')})})
    if version!='legacy':review['initial_annotation_sha256']=old_hash
    write(tmp_path/'input.json',review);import_review(tmp_path,tmp_path/'input.json')
    agreement=read(next((tmp_path/'reviews').glob('*.agreement.json')))
    assert agreement['agreement']==(1.0 if version=='historical' else None)
    assert agreement['comparison_status']=={'historical':'matched_initial_version','missing':'initial_version_unavailable','legacy':'unversioned_review'}[version]


@pytest.mark.parametrize('pending',[40,3,0])
def test_verifier_progress_is_dynamic(pending):
    """50题完成后不得继续报告剩余40题，也不据此宣称全部研究完成。"""
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/verify_evidence_audit.py'
    spec=importlib.util.spec_from_file_location('audit_verifier_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    status,incomplete=module.review_progress({'pending_questions':pending})
    if pending:assert incomplete[0]==f'其余{pending}题逐帧初审' and status=='initial_review_incomplete'
    else:assert not any('题逐帧初审' in x for x in incomplete) and status=='initial_review_complete_other_checks_pending'
