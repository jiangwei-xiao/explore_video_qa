"""定位文本探针的纯接口检查，不调用模型或读取标准答案。"""
import pytest
from videoqa_audit.query_input_probe import build_query,ranking


def test_query_uses_all_options_in_original_order():
    """问题、限定、否定、数字及各选项原样保留，不选择某个答案。"""
    q='Which event happened before 1613?';options=['A. A','B. B','C. Not C','D. D']
    assert build_query(q,options)==q+'\n'+'\n'.join(options)
    assert options==['A. A','B. B','C. Not C','D. D']


@pytest.mark.parametrize('q,options',[('',list('ABCD')),('q',list('ABC')),('q',['A','B',None,'D'])])
def test_bad_input_rejected(q,options):
    """接口：缺失文本不可静默截断或补造。"""
    with pytest.raises(ValueError):build_query(q,options)


def test_rank_ties():
    """同分先PTS后编号，与历史Top-K一致。"""
    c=[dict(source_pts=20,candidate_index=0),dict(source_pts=10,candidate_index=1),dict(source_pts=10,candidate_index=2)]
    assert ranking(c,[.5,.5,.5])==[1,2,0]


@pytest.mark.parametrize('scores',[[float('nan')],[1.1],[-.1],[]])
def test_bad_scores_rejected(scores):
    """接口：评分范围及数量严格校验。"""
    with pytest.raises(ValueError):ranking([dict(source_pts=0,candidate_index=0)],scores)


def fake_batch(tmp_path,fail=False):
    """接口：只用CPU张量和假评分器测试事务，不加载任何神经模型。"""
    import torch
    import numpy as np
    from videoqa_full.state import durable
    durable(tmp_path/'protocol.json',dict(test=True))
    class Fake:
        calls=0
        def score_batch_features(self,encoded,images):
            self.calls+=1
            if fail:raise RuntimeError('simulated interrupted scoring')
            return [.3],np.ones((1,2),dtype=np.float32),.01,.02
    return Fake(),{'input_ids':torch.tensor([[101,102]])}


def test_completed_batch_resumes_without_forward(tmp_path):
    """完成批次只复用落盘返回，不重复评分。"""
    from videoqa_audit.query_input_probe import score_once
    scorer,encoded=fake_batch(tmp_path)
    for _ in range(2):assert score_once(tmp_path,'q','question_only',0,scorer,encoded,[None],'0',['rgb'])[0]==[.3]
    assert scorer.calls==1


def test_uncertain_batch_does_not_repeat(tmp_path):
    """异常已开始事务必须暂停，恢复不能再次执行前向。"""
    from videoqa_audit.query_input_probe import score_once
    scorer,encoded=fake_batch(tmp_path,True)
    with pytest.raises(RuntimeError):score_once(tmp_path,'q','question_only',0,scorer,encoded,[None],'0',['rgb'])
    with pytest.raises(AssertionError):score_once(tmp_path,'q','question_only',0,scorer,encoded,[None],'0',['rgb'])
    assert scorer.calls==1


def test_changed_rgb_is_not_resumed(tmp_path):
    """像素身份变更时拒绝读取旧评分。"""
    from videoqa_audit.query_input_probe import score_once
    scorer,encoded=fake_batch(tmp_path)
    score_once(tmp_path,'q','question_only',0,scorer,encoded,[None],'0',['rgb'])
    with pytest.raises(AssertionError):score_once(tmp_path,'q','question_only',0,scorer,encoded,[None],'0',['changed'])
    assert scorer.calls==1
