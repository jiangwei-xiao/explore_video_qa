"""全量验证的预算、原始返回恢复、输入身份和冻结写入边界。"""
from pathlib import Path
import copy
import pytest
from videoqa_full.state import durable
from videoqa_runtime.common import sha256, read_json
from videoqa_generalization.inputs import frozen
from videoqa_generalization.run import STAGES, recover
from videoqa_generalization import llava_observe
from test_retarget_seed_pool import fixture
from videoqa_methods.retarget_seed_pool import retarget_pool
from videoqa_methods.seed_budget_selection import select_seed_budget


def test_m16_selection_uses_rebuilt_pool_and_preserves_original():
    """小型独立池验证改变M/K后仍保护锚点、保持源序、特征映射且不改旧池。"""
    pool, features, cfg = fixture(); before=copy.deepcopy(pool)
    narrow, nf=retarget_pool(pool,features,dict(cfg,seed_budget=4,frame_budget=6))
    indices,trace=select_seed_budget(narrow,nf,frame_budget=6)
    assert len(indices)==len(set(indices))==6 and set(trace['protected_indices'])<=set(indices)
    pts=[narrow['candidates'][i]['source_pts'] for i in indices]
    assert pts==sorted(pts) and pool==before
    assert [pool['candidates'][narrow['original_candidate_indices'][i]]['source_pts'] for i in indices]==pts


def test_frozen_asset_never_replaces_old_input(tmp_path):
    p=tmp_path/'input.json'; frozen(p,{'pts':[1,2]}); before=sha256(p)
    frozen(p,{'pts':[1,2]})
    with pytest.raises(ValueError,match='Frozen asset'):frozen(p,{'pts':[1,3]})
    assert sha256(p)==before


def test_real_call_matrix_and_import_budget():
    assert sum(spec[3] for spec in STAGES.values())==7900
    assert STAGES['qwen16'][2:]==STAGES['llava16'][2:]==(16,2700)
    assert STAGES['wfs32'][2:]==(32,2500)


def test_unconfirmed_llava_generation_cannot_be_reconstructed():
    with pytest.raises(ValueError,match='not been confirmed'):
        llava_observe.result_from_raw({'generation_state':{'started':True}}, {})


def test_unknown_call_without_raw_never_reenters_generation(tmp_path):
    durable(tmp_path/'protocol.json',{'model':'llava'})
    durable(tmp_path/'calls/RD-1.2-exp12/q.json',dict(status='failed_or_uncertain',
        protocol_sha256=sha256(tmp_path/'protocol.json'),gpu='1'))
    with pytest.raises(FileNotFoundError):recover(tmp_path,{'question_id':'q'},dict(methods=['RD-1.2-exp12'],model_kind='llava'))


def test_confirmed_return_restores_result_without_model(tmp_path,monkeypatch):
    """故障发生在返回后时，仅恢复原始文本计分和结果，不要求GPU或模型对象。"""
    class Task:
        def videomme_process_results(self,doc,answers):
            return {'videomme_perception_score':{'pred_answer':answers[0],'answer':'B'}}
    monkeypatch.setattr(llava_observe,'load_task',lambda:Task())
    monkeypatch.setattr(llava_observe,'task_doc',lambda row:row)
    method='RD-1.2-exp12'; row=dict(question_id='q',video_id='v',question='Q',options=['A','B'],stratum='short')
    durable(tmp_path/'protocol.json',{'model':'LLaVA'})
    fp=sha256(tmp_path/'protocol.json'); identity=dict(question_id='q',method=method,protocol_sha256=fp,gpu='1')
    durable(tmp_path/'calls'/method/'q.json',dict(status='failed_or_uncertain',protocol_sha256=fp,gpu='1'))
    durable(tmp_path/'raw_generations'/method/'q.json',dict(identity=identity,
        generation_state={'started':True,'returned':True},raw_generated_text=' B ',context='Q',prompt='Q',observed={}))
    durable(tmp_path/'selections'/method/'q.json',dict(selection={'selected_frames':[]},rgb_sha256=[]))
    recover(tmp_path,row,dict(methods=[method],model_kind='llava',model='LLaVA'))
    result=read_json(tmp_path/'results'/method/'q.json')
    assert result['correct'] and result['answer']['raw_output']=='B' and result['timing_kind']=='resumed_not_primary'
    before=sha256(tmp_path/'raw_generations'/method/'q.json')
    recover(tmp_path,row,dict(methods=[method],model_kind='llava',model='LLaVA'))
    assert sha256(tmp_path/'raw_generations'/method/'q.json')==before
