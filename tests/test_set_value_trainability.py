"""CPU诊断的真实边界：重放/恢复、留出隔离、唯一输入及训练停止规则。"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch

from videoqa_audit.set_value_trainability import (
    SPEC, TrainingAssets, aggregate_cases, describe_case, freeze_folds,
    parameter_digest, passed_fit, select_methods, state_error, train_trajectory,
)


def fixture_data(n=8):
    """接口：人工数值资产用于测试优化数据流，不代表实际问答或研究结论。"""
    rng=np.random.default_rng(2027)
    x=rng.normal(size=(n,4,8,770)).astype(np.float32)
    y=np.tile(np.array([1,0,1,0],dtype=np.float32),(n,1))
    for j in range(4):x[:,j,:,0]=(1 if j in (0,2) else -1)
    rows=[dict(question_id=f'q{i}',video_id=f'v{i//2}',split='train') for i in range(n)]
    return TrainingAssets(rows,x,y,[[0,1,2,3] for _ in range(n)],[[[[j,0]] for j in range(4)] for _ in range(n)])


def small_spec(**kwargs):
    """仅测试缩小更新上限；正式入口始终固定SPEC。"""
    return dict(SPEC,max_updates=6,check_every=2,required_accuracy=1.1,**kwargs)


def test_fold_has_two_questions_per_video_and_fixed_preference_sizes():
    rows=[dict(question_id=f'q{i}',video_id=f'v{i//2}',split='train') for i in range(600)]
    a=freeze_folds(rows);b=freeze_folds(list(rows))
    assert a==b
    assert sorted(i for f in a['folds'] for i in f['heldout_indices'])==list(range(600))
    for fold in a['folds']:
        train={rows[i]['video_id'] for i in fold['train_indices']}
        held={rows[i]['video_id'] for i in fold['heldout_indices']}
        assert len(train)==240 and len(held)==60 and not train&held


def test_validation_rows_cannot_enter_training_or_folds():
    data=fixture_data();rows=deepcopy(data.rows);rows[0]['split']='val'
    with pytest.raises(ValueError,match='training rows'):
        TrainingAssets(rows,data.features,data.labels,data.representatives,data.source_orders)
    with pytest.raises(ValueError,match='Validation rows'):
        freeze_folds(rows)


def test_input_identity_dedup_does_not_fabricate_votes():
    frames=[{(0,1),(1,2)},{(0,1),(1,2)},{(2,3),(3,4)},{(4,5),(5,6)}]
    c=describe_case('q','v','T','train',30.,[1,1,0,0],['a','a','b','c'],frames)
    assert c['unique_inputs']==3 and c['preference_pairs']==2 and c['complete_input_equal_pairs']==1
    stats=aggregate_cases([c]);assert stats['unique_preference_pairs']==2 and stats['mixed_fraction']==1.
    conflict=describe_case('q','v','T','train',30.,[1,0,0,0],['a','a','b','c'],frames)
    assert conflict['complete_input_conflict'] and conflict['preference_pairs']==0


@pytest.mark.parametrize('y,state',[([1,1,1,1],'all_correct'),([0,0,0,0],'all_wrong')])
def test_all_equal_answers_make_no_preferences(y,state):
    c=describe_case('q','v','T','train',30.,y,list('abcd'),[{(i,i)} for i in range(4)])
    assert c['state']==state and c['preference_pairs']==0


def test_stopping_uses_both_training_measures():
    assert passed_fit(dict(preference_accuracy=.96,usable_top1_accuracy=.96),SPEC)
    assert not passed_fit(dict(preference_accuracy=.94,usable_top1_accuracy=1.),SPEC)
    assert not passed_fit(dict(preference_accuracy=1.,usable_top1_accuracy=.94),SPEC)


def test_tie_order_uses_sources_not_answers():
    assert select_methods(np.zeros((1,4)),[[[[4,0]],[[2,0]],[[3,0]],[[1,0]]]])==[3]


def test_optimizer_comparison_includes_moments_and_step():
    a=dict(state={0:dict(step=torch.tensor(40.),exp_avg=torch.ones(3))},param_groups=[dict(lr=.001)])
    assert state_error(a,deepcopy(a))==0.
    b=deepcopy(a);b['state'][0]['exp_avg'][0]+=1
    assert state_error(a,b)==1.
    b=deepcopy(a);b['param_groups'][0]['lr']=.01
    with pytest.raises(ValueError):state_error(a,b)


def test_resume_continues_after_committed_step_without_double_update(tmp_path):
    torch.set_num_threads(2);data=fixture_data();ids=[r['question_id'] for r in data.rows];spec=small_spec()
    one=train_trajectory(data,tmp_path/'one',ids,spec=spec)
    paused=train_trajectory(data,tmp_path/'resumed',ids,spec=spec,stop_after=3)
    assert paused==dict(status='interrupted_for_test',step=3)
    two=train_trajectory(data,tmp_path/'resumed',ids,spec=spec)
    a=torch.load(tmp_path/'one/state.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'resumed/state.pt',map_location='cpu',weights_only=False)
    assert one['updates']==two['updates']==6
    assert state_error(a['model'],b['model'])==state_error(a['optimizer'],b['optimizer'])==0.
    before=(tmp_path/'resumed/state.pt').read_bytes()
    assert train_trajectory(data,tmp_path/'resumed',ids,spec=spec)==two
    assert (tmp_path/'resumed/state.pt').read_bytes()==before


def test_fold_training_objects_do_not_contain_holdout_labels(tmp_path):
    data=fixture_data();changed=deepcopy(data);changed.labels[4:]=1-changed.labels[4:]
    train_a=data.subset([0,1,2,3]);train_b=changed.subset([0,1,2,3])
    ids=[r['question_id'] for r in train_a.rows]
    train_trajectory(train_a,tmp_path/'a',ids,spec=small_spec())
    train_trajectory(train_b,tmp_path/'b',ids,spec=small_spec())
    a=torch.load(tmp_path/'a/state.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'b/state.pt',map_location='cpu',weights_only=False)
    assert state_error(a['model'],b['model'])==0.
    assert a['initialization_sha256']==b['initialization_sha256']


def test_wrong_fold_membership_rejected(tmp_path):
    data=fixture_data()
    with pytest.raises(ValueError,match='membership'):
        train_trajectory(data,tmp_path/'bad',['wrong']*8,spec=small_spec())


def test_actual_40_step_control_and_continuation(tmp_path):
    """数值fixture只检查40步比对和继续分支，不加载任何骨干模型。"""
    data=fixture_data();ids=[r['question_id'] for r in data.rows]
    spec=dict(SPEC,max_updates=40,required_accuracy=1.1)
    base=train_trajectory(data,tmp_path/'reference',ids,spec=spec)
    checkpoint=torch.load(tmp_path/'reference/state.pt',map_location='cpu',weights_only=False)
    result=train_trajectory(data,tmp_path/'check',ids,spec=dict(spec,max_updates=42),reference=checkpoint)
    assert result['updates']==42 and result['replay_40']['model_abs_error']==0.
    assert result['replay_40']['optimizer_abs_error']==0.
    assert result['initialization_sha256']==base['initialization_sha256']


def test_fit_requires_three_consecutive_training_checks(tmp_path):
    data=fixture_data();ids=[r['question_id'] for r in data.rows]
    spec=dict(SPEC,max_updates=20,check_every=2,required_accuracy=0.)
    result=train_trajectory(data,tmp_path/'stable',ids,spec=spec)
    assert result['status']=='fit_passed' and result['updates']==6
    assert [p['step'] for p in result['history']]==[0,2,4,6]


def test_corrupted_optimizer_reference_fails_40_step_control(tmp_path):
    data=fixture_data();ids=[r['question_id'] for r in data.rows]
    spec=dict(SPEC,max_updates=40,required_accuracy=1.1)
    train_trajectory(data,tmp_path/'old',ids,spec=spec)
    checkpoint=torch.load(tmp_path/'old/state.pt',map_location='cpu',weights_only=False)
    next(iter(checkpoint['optimizer']['state'].values()))['step']+=1
    with pytest.raises(ValueError,match='40-step replay'):
        train_trajectory(data,tmp_path/'wrong',ids,spec=spec,reference=checkpoint)
