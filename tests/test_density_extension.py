"""扩展题集与配对执行边界测试；不加载模型或运行问答。"""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from videoqa_audit.density_extension import check_recovery,verify_pair

path=Path(__file__).resolve().parents[1]/'scripts/prepare_density_extension.py'
spec=importlib.util.spec_from_file_location('extension_sampling',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def synthetic_rows():
    """构造足够候选视频，每视频三题，测试抽样只依赖身份字段。"""
    return [dict(video_id=f'{s}_{i}',question_id=f'{s}_{i}-{j}',stratum=s,answer_index=j%4)
            for s in ('short','medium','long') for i in range(80) for j in range(3)]


def test_sampling_is_video_disjoint_and_balanced():
    rows=synthetic_rows();old=rows[:3];reserve=rows[3:6]
    selected=module.select_new_rows(rows,old,reserve)
    assert len(selected)==150 and len({r['video_id'] for r in selected})==150
    assert all(sum(r['stratum']==s for r in selected)==50 for s in ('short','medium','long'))
    assert not {r['video_id'] for r in selected}&{r['video_id'] for r in old+reserve}


def test_sampling_ignores_answers_and_row_order():
    rows=synthetic_rows();a=module.select_new_rows(rows,[],[])
    changed=copy.deepcopy(rows[::-1])
    for r in changed:r.update(answer_index=99,correct=False,score=999)
    b=module.select_new_rows(changed,[],[])
    assert [r['question_id'] for r in a]==[r['question_id'] for r in b]


def test_sampling_refuses_insufficient_video_pool():
    with pytest.raises(AssertionError):module.select_new_rows(synthetic_rows()[:30],[],[])


def test_recovery_refuses_uncertain_generation(tmp_path):
    directory=tmp_path/'calls/answer';directory.mkdir(parents=True)
    (directory/'x__v1.json').write_text(json.dumps(dict(status='failed_or_uncertain',protocol_sha256='f')))
    with pytest.raises(AssertionError):check_recovery(tmp_path,[{'question_id':'x'}],'f')


def pair_fixture(tmp_path):
    """最小上游身份和特征文件，结果正确性不参与输入一致性判断。"""
    s={k:[] for k in ('video','candidates','plan','refinement')}
    s.update(initial_count=2,new_candidate_count=0,initial_scores=[.2,.7])
    for v in ('v1','soft'):
        d=tmp_path/'results'/v;d.mkdir(parents=True)
        np.save(tmp_path/f'{v}.npy',np.eye(2,dtype=np.float32))
        (d/'x.json').write_text(json.dumps(dict(selection=s,feature_file=f'{v}.npy')))


def test_pair_accepts_identical_independent_pools(tmp_path):
    pair_fixture(tmp_path)
    result=verify_pair(tmp_path,'x')
    assert result['score_max_abs_error']==0 and result['feature_max_abs_error']==0


def test_pair_refuses_feature_drift(tmp_path):
    pair_fixture(tmp_path);np.save(tmp_path/'soft.npy',np.ones((2,2),dtype=np.float32))
    with pytest.raises(AssertionError):verify_pair(tmp_path,'x')
