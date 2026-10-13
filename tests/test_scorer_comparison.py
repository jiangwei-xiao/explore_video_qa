"""评分器替换回归：纯视觉/ITM隔离、预算、同输入共享与后台阶段顺序。"""
from types import SimpleNamespace
import numpy as np
import pytest
from videoqa_methods.retrieval_frontends import normalize_cls,Blip2ItmFrontend,REVISION
from videoqa_scorer_comparison.runtime import group_inputs,observed_identity
from videoqa_scorer_comparison.preparation import select_pool,source_keys,METHODS


@pytest.mark.parametrize('value',[[],[1,2],[[0.,0.]],[[float('nan'),1]],[[float('inf'),1]]])
def test_cls_invalid(value):
    """接口：非二维、非有限和零向量不能作为相似度输入。"""
    with pytest.raises(ValueError):normalize_cls(value)


def test_cls_normalization():
    """接口：实际维度可变化，归一化保留纯视觉方向与FP32。"""
    x=normalize_cls([[3,4,0],[0,0,8]])
    assert x.dtype==np.float32
    assert np.allclose(np.linalg.norm(x,axis=1),1)
    assert np.allclose(x[0],[.6,.8,0])


def test_untruncated_position_limit():
    """接口：文本超过QFormer实际上限即拒绝，不沿用32Token隐式截断。"""
    calls=[];frontend=Blip2ItmFrontend.__new__(Blip2ItmFrontend);frontend.limit=64
    def tokenize(text,**kw):
        calls.append(kw);return {'input_ids':np.zeros((1,len(text)))}
    frontend.processor=SimpleNamespace(tokenizer=tokenize)
    assert frontend.tokenize('x'*64)['input_ids'].shape==(1,64)
    with pytest.raises(ValueError):frontend.tokenize('x'*65)
    assert all(c['truncation'] is False for c in calls)


def test_group_import_does_not_depend_on_score():
    """接口：历史引用匹配来自完整输入，同组某方法可提供已经验证的物理来源。"""
    items={m:dict(input_identity={'pixels':'a','prompt':'same'}) for m in METHODS}
    items[METHODS[2]].update(historical_result='source.json',historical_sources={'source.json':'hash'})
    g=group_inputs(items)
    assert len(g)==1 and g[0]['representative']==METHODS[0]
    assert g[0]['historical_result']=='source.json' and g[0]['methods']==list(METHODS)


@pytest.mark.parametrize('field',['pixels','prompt','tokens','source'])
def test_distinct_input_not_shared(field):
    """接口：任一输入身份不同均不能引用，禁止仅凭源编号简化判断。"""
    a={'pixels':'a','prompt':'p','tokens':[1],'source':[1]};b=dict(a);b[field]='different'
    items={m:dict(input_identity=dict(a)) for m in METHODS};items[METHODS[2]]['input_identity']=b
    assert len(group_inputs(items))==2


def toy_pool(n=6,k=4):
    """接口：短池/同分/两版共享前缀的确定性输入，不包含答案信息。"""
    from videoqa_methods.budget_contract import budget_config
    from videoqa_methods.density_selection import build_regions
    video=dict(start_pts=0,time_base='1/4',duration_seconds=n+1,duration_fraction=str(n+1),path='synthetic.mp4')
    frames=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,
        timestamp_seconds=i+.25,source_seconds=i+.25,requested_seconds=i+.25) for i in range(n)]
    scores=[.5]*n;plan=build_regions(frames,scores,video,budget_config(k))
    return dict(video=video,candidates=frames,initial_count=n,initial_scores=scores,plan=plan),np.eye(n,dtype=np.float32)


@pytest.mark.parametrize('n,k',[(2,8),(6,4),(8,8)])
def test_budget_and_uniform_coverage_equivalence(n,k):
    """接口：短池不凑数；等分数exp07/12的覆盖目标等价，保护锚点且唯一有序。"""
    p,f=toy_pool(n,k);selections,trace=select_pool(p,f,k)
    assert all(len(v)==min(n,k) for v in selections.values())
    assert source_keys(selections[METHODS[1]])==source_keys(selections[METHODS[2]])
    for frames in selections.values():
        keys=source_keys(frames);assert len(set(keys))==len(keys)
        assert [a[1] for a in keys]==sorted(a[1] for a in keys)
    assert sum(trace['exp07']['region_budgets'].values())==min(n,k)


def test_model_revision_fixed():
    """接口：使用ITM权重固定revision，不接受main漂移。"""
    assert REVISION=='520bf73fd0ef6ccce791cd3f06908040aef2d8cc'


def test_observed_identity_qwen_padding_fields():
    """接口：实际Qwen视频grid不能在同帧引用时遗漏。"""
    answer={'observed':dict(prompt='p',input_token_ids=[1,2],processor_pixel_sha256='h',video_grid_thw=[[6,16,16]])}
    assert observed_identity('qwen32',answer)['video_grid_thw']==[[6,16,16]]


def test_itm_uses_positive_probability_and_pure_vision(monkeypatch):
    """接口：假模型区分ITM/ITC及纯视觉字段，确保接口不误取QFormer融合特征。"""
    import torch
    from PIL import Image
    frontend=Blip2ItmFrontend.__new__(Blip2ItmFrontend);frontend.torch=torch
    frontend.counts=dict(itm_batches=0,itm_frames=0,visual_batches=0,visual_frames=0)
    original_to=torch.Tensor.to
    def to(self,*args,**kw):
        if args and args[0]=='cuda':args=('cpu',)+args[1:]
        return original_to(self,*args,**kw)
    monkeypatch.setattr(torch.Tensor,'to',to);monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    frontend.processor=lambda **kw:dict(pixel_values=torch.ones(len(kw['images']),3,4,4))
    def forward(**kw):
        assert kw['use_image_text_matching_head'] is True
        return SimpleNamespace(logits_per_image=torch.tensor([[0.,2.],[2.,0.]]),
            vision_model_output=SimpleNamespace(last_hidden_state=torch.tensor([[[3.,4.]],[[0.,2.]]])),
            image_embeds=torch.full((2,1,2),99.))
    frontend.model=forward
    images=[Image.new('RGB',(4,4)) for _ in range(2)]
    score,f,_,_=frontend.score_batch_features(dict(input_ids=torch.ones(1,3,dtype=torch.long)),images)
    assert score[0]>.8 and score[1]<.2
    assert np.allclose(f,[[.6,.8],[0.,1.]])
    assert frontend.counts['itm_frames']==2


def test_existing_qwen_input_import_smoke(tmp_path):
    """接口：真实历史一题三输入只经CPU，核验视频/RGB/像素/完整提示与物理来源。"""
    import sys
    if 'lmms_qwen' not in sys.executable:pytest.skip('Uses isolated Qwen processor environment')
    from videoqa_full.state import rows,durable
    from videoqa_runtime.common import read_json
    from videoqa_audit.framework_contraction import RUN32
    from videoqa_audit.region_scope_runtime import parent_result
    from videoqa_scorer_comparison import runtime as rt
    row=rows()[0];profile='qwen32';q=row['question_id']
    durable(tmp_path/'qa'/profile/'parent_protocol.json',read_json(RUN32/'protocol.json'))
    for m in METHODS:
        old=read_json(parent_result(profile,q,m))
        durable(tmp_path/'selections'/profile/m/(q+'.json'),dict(selection=old['selection']))
    rt._TOOLS=rt.make_tools(profile)
    prepared=rt.prepare_question((str(tmp_path),profile,row))
    assert all(g['historical_result'] for g in prepared['groups'])
    assert sum(len(g['methods']) for g in prepared['groups'])==3


@pytest.mark.parametrize('raw_state',[{'started':True,'returned':False},{'started':False,'returned':False}])
def test_uncertain_generation_not_repeated(tmp_path,raw_state):
    """接口：未返回的原始状态不能由恢复入口变成第二次生成。"""
    from videoqa_full.state import durable
    from videoqa_runtime.common import sha256
    from videoqa_scorer_comparison.runtime import recover_group
    durable(tmp_path/'protocol.json',{'profile':'qwen32'})
    fp=sha256(tmp_path/'protocol.json');m=METHODS[0];q='q';gpu='3'
    durable(tmp_path/'calls'/m/(q+'.json'),dict(status='failed_or_uncertain',protocol_sha256=fp,gpu=gpu))
    durable(tmp_path/'raw_generations'/m/(q+'.json'),dict(identity=dict(question_id=q,method=m,protocol_sha256=fp,gpu=gpu),generation_state=raw_state))
    with pytest.raises(ValueError,match='Uncertain'):
        recover_group(tmp_path,{'question_id':q},{'profile':'qwen32'},dict(representative=m,methods=list(METHODS)))
    assert not (tmp_path/'results').exists()


def test_returned_generation_restores_shared_results_without_inference(tmp_path,monkeypatch):
    """接口：一次完整返回可恢复三份引用，第二次恢复不再解析或调用生成。"""
    from videoqa_full.state import durable
    from videoqa_runtime.common import sha256,read_json
    from videoqa_scorer_comparison.runtime import recover_group
    import videoqa_budget32.observe as obs
    calls=[];answer=dict(parsed_answer='A',raw_output='A')
    monkeypatch.setattr(obs,'result_from_raw',lambda raw,row:calls.append(raw) or answer)
    spec=dict(profile='qwen32',model='frozen');durable(tmp_path/'protocol.json',spec)
    fp=sha256(tmp_path/'protocol.json');m=METHODS[0];q='q';gpu='3'
    row=dict(question_id=q,video_id='v',question='original',options=['a','b','c','d'],stratum='short',answer_index=0)
    durable(tmp_path/'calls'/m/(q+'.json'),dict(status='failed_or_uncertain',protocol_sha256=fp,gpu=gpu))
    durable(tmp_path/'raw_generations'/m/(q+'.json'),dict(identity=dict(question_id=q,method=m,protocol_sha256=fp,gpu=gpu),generation_state={'started':True,'returned':True}))
    for method in METHODS:durable(tmp_path/'selections'/method/(q+'.json'),dict(selection={'source':'same'},rgb_sha256=['same']))
    group=dict(representative=m,methods=list(METHODS),historical_sources={})
    recover_group(tmp_path,row,spec,group);recover_group(tmp_path,row,spec,group)
    assert len(calls)==1
    assert len(list((tmp_path/'results').glob('*/*.json')))==3
    assert read_json(tmp_path/'calls'/m/(q+'.json'))['status']=='completed'


def test_proxy_rank_and_exact_window_endpoints():
    """接口：事实仅用于评价；同分排序及真实PTS闭窗口端点正确。"""
    from videoqa_scorer_comparison.analysis import stable_ranks,witness_status
    pool,_=toy_pool(6,4)
    ranks=stable_ranks(pool['candidates'],pool['initial_scores'])
    assert ranks[pool['candidates'][0]['source_pts']]==1
    pool['plan']['regions']=[dict(left_fraction='1/4',right_fraction='5/4')]
    frames=[pool['candidates'][1]]
    value=witness_status(pool,frames,[pool['candidates'][1]['source_pts']])
    assert value['window_reachable'] is True and value['selected_exact'] is True
    assert value['best_rank']==2


def test_proxy_missing_witness_not_silently_removed():
    """接口：主75事实若无法映射原粗池，应暂停而非自动缩小分母。"""
    from videoqa_scorer_comparison.analysis import witness_status
    pool,_=toy_pool()
    with pytest.raises(ValueError,match='missing'):
        witness_status(pool,[],[99999])
