"""密度框架模块验收：不加载真实模型、不使用真实题答案。"""
from copy import deepcopy
from fractions import Fraction
import math
import numpy as np
import pytest
from videoqa_runtime.baseline_selection import ProtocolError, iter_candidate_frames
from videoqa_runtime.common import read_json, ROOT, write_json
from videoqa_methods.density_selection import build_regions, refinement_targets, select_joint, source_time
from videoqa_methods.density_pipeline import decode_regions, prepare_density
from videoqa_methods.density_run import recovery_check
from videoqa_methods.followups import CallLedger

CFG=read_json(ROOT/'configs/retrieval_density_v1.json')


def pool(n=40, origin=0):
    """构造带非零起点的精确时钟，目标与源时间分开保存。"""
    rows=[dict(candidate_index=i,source_pts=origin+4*i+1,source_frame_index=4*i+1,
               source_seconds=(origin+4*i+1)/4,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(n)]
    return rows,[1-i/(n+1) for i in range(n)],dict(time_base='1/4',start_pts=origin,duration_seconds=float(n),duration_fraction=str(n))


@pytest.mark.parametrize('ids,counts,weight,enabled',[
    ([0],[1],1,False),([0,1,2],[3],3,True),([0,1,4,5],[2,2],3,False),
    ([0,1,2,5],[3,1],3,True),([0,3,6],[1,1,1],1,False)])
def test_core_counts_and_trigger(ids,counts,weight,enabled):
    rows,scores,video=pool();scores=[.8 if i in ids else .1 for i in range(len(rows))]
    cfg=dict(CFG,seed_budget=len(ids));plan=build_regions(rows,scores,video,cfg)
    assert len(plan['regions'])==1
    r=plan['regions'][0]
    assert r['core_seed_counts']==counts and r['weight']==weight and r['refine']==enabled
    assert r['anchor_index']==min(ids)


def test_core_connection_uses_requested_time_not_pts_jitter():
    rows,scores,video=pool();video.update(time_base='1/1000',start_pts=0)
    for i,row in enumerate(rows):
        row.update(source_pts=250+i*1000,timestamp_seconds=.25+i)
    rows[2].update(source_pts=2252,timestamp_seconds=2.252)
    scores=[.9 if i in (0,2,5) else .1 for i in range(len(rows))]
    plan=build_regions(rows,scores,video,dict(CFG,seed_budget=3))
    assert [c['seed_indices'] for c in plan['cores']]==[[0,2],[5]]


def test_touching_windows_merge_without_merging_original_cores():
    rows,scores,video=pool(origin=1000);scores=[.9 if i in (0,8) else .1 for i in range(len(rows))]
    plan=build_regions(rows,scores,video,dict(CFG,seed_budget=2))
    assert len(plan['cores'])==2 and len(plan['regions'])==1
    assert plan['regions'][0]['left_fraction']=='0'
    assert plan['regions'][0]['right_fraction']=='49/4'
    assert plan['regions'][0]['weight']==1 and not plan['regions'][0]['refine']
    assert source_time(rows[0],video)==Fraction(1,4)


@pytest.mark.parametrize('left,right,expected',[
    ('0','4',['3/4','7/4','11/4','15/4']),('3/4','7/4',['3/4','7/4']),
    ('0','1/2',[]),('157/4','40',['159/4'])])
def test_refine_grid_endpoints(left,right,expected):
    _,_,video=pool()
    assert list(map(str,refinement_targets(dict(left_fraction=left,right_fraction=right),video,CFG)))==expected


@pytest.mark.parametrize('n',[1,3,11,16])
def test_effective_seed_and_frame_counts(n):
    rows,scores,video=pool(n);plan=build_regions(rows,scores,video,CFG)
    features=np.tile(np.array([1.,0.],np.float32),(n,1))
    selected,trace=select_joint(rows,features,plan,video,CFG)
    assert plan['effective_seed_count']==n and len(selected)==n
    assert selected==list(range(n)) and set(trace['protected_indices'])<=set(selected)


def test_sixteen_separate_regions_naturally_return_topk():
    rows,scores,video=pool(170);ids=list(range(0,160,10))
    scores=[.9 if i in ids else .1 for i in range(len(rows))]
    plan=build_regions(rows,scores,video,CFG);assert len(plan['regions'])==16
    features=np.tile(np.array([1.,0.],np.float32),(len(rows),1))
    selected,trace=select_joint(rows,features,plan,video,CFG)
    assert selected==ids and trace['rounds']==[]


def test_all_selected_frames_are_redundancy_references():
    rows,scores,video=pool(4);cfg=dict(CFG,frame_budget=3)
    plan=build_regions(rows,scores,video,cfg)
    features=np.array([[1,0,0],[0,1,0],[0,.99,.141],[.5,.5,.707]],np.float32)
    features/=np.linalg.norm(features,axis=1,keepdims=True)
    chosen,trace=select_joint(rows,features,plan,video,cfg)
    assert chosen==[0,1,3]
    assert trace['rounds'][0]['winner']['candidate_index']==1


def test_no_cross_region_visual_dedup_and_source_labels_neutral():
    rows,scores,video=pool(50);scores=[.9 if i in (0,30) else .1 for i in range(50)]
    cfg=dict(CFG,seed_budget=2,frame_budget=4);plan=build_regions(rows,scores,video,cfg)
    features=np.tile(np.array([1.,0.],np.float32),(50,1));features[[1,31]]=[0,1]
    chosen,trace=select_joint(rows,features,plan,video,cfg)
    assert chosen==[0,1,30,31]
    changed=[dict(r,origin='new' if i%2 else 'coarse') for i,r in enumerate(rows)]
    assert select_joint(changed,features,plan,video,cfg)==(chosen,trace)


def test_zero_gain_fill_and_tie_are_deterministic():
    rows,scores,video=pool(20);scores=[.5]*20;scores[5]=1.;cfg=dict(CFG,frame_budget=5)
    plan=build_regions(rows,scores,video,cfg);features=np.tile(np.array([1.,0.],np.float32),(20,1))
    chosen,trace=select_joint(rows,features,plan,video,cfg)
    assert chosen==[0,1,2,3,5]
    assert all(t['zero_diversity_fill'] for t in trace['rounds'])


def test_weight_is_not_a_fixed_quota():
    rows,scores,video=pool(50);scores=[.9 if i in (0,1,2,30) else .1 for i in range(50)]
    cfg=dict(CFG,seed_budget=4,frame_budget=3);plan=build_regions(rows,scores,video,cfg)
    features=np.tile(np.array([1.,0.],np.float32),(50,1));features[31]=[0,1]
    chosen,trace=select_joint(rows,features,plan,video,cfg)
    assert chosen==[0,30,31] and trace['region_budgets']=={'0':1,'1':2}


@pytest.mark.parametrize('defect',['nan','not_normalized','duplicate','empty','capacity'])
def test_invalid_inputs_stop(defect):
    rows,scores,video=pool(20);cfg=dict(CFG);features=np.tile(np.array([1.,0.],np.float32),(20,1))
    if defect=='nan':scores[0]=float('nan')
    if defect=='duplicate':rows[1]['source_pts']=rows[0]['source_pts']
    if defect=='empty':rows=[];scores=[]
    if defect=='capacity':
        scores=[.9 if i in (0,19) else .1 for i in range(20)];cfg.update(seed_budget=2,frame_budget=1)
    if defect=='not_normalized':features[0]=[2,0]
    with pytest.raises(ProtocolError):
        plan=build_regions(rows,scores,video,cfg);select_joint(rows,features,plan,video,cfg)


def make_video(path, frames=360, vfr=True):
    """生成小尺寸测试视频，保留精确PTS以核对局部解码帧号。"""
    import av
    with av.open(str(path),'w') as output:
        stream=output.add_stream('mpeg4',rate=8);stream.width=64;stream.height=48;stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,8)
        for i in range(frames):
            frame=av.VideoFrame.from_ndarray(np.full((48,64,3),i%255,np.uint8),format='rgb24')
            frame.pts=i+(i//17 if vfr else 0);frame.time_base=Fraction(1,8)
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    metadata={};rows=[r for r,f in iter_candidate_frames(path,metadata)]
    with av.open(str(path)) as source:
        stream=source.streams.video[0];metadata['duration_fraction']=str(stream.duration*Fraction(stream.time_base))
        exact={f.pts:i for i,f in enumerate(source.decode(stream))}
    return rows,metadata,exact


def test_vfr_decode_new_frames_exact_and_not_old_24_cap(tmp_path):
    rows,video,exact=make_video(tmp_path/'vfr.mp4')
    scores=[.9 if i in range(0,32,2) else .1 for i in range(len(rows))]
    plan=build_regions(rows,scores,video,CFG)
    added,assets,reports,_=decode_regions(video,rows,plan,CFG,tmp_path/'images')
    assert len(added)>24 and len(assets)==len(added)
    assert not ({r['source_pts'] for r in rows}&{r['source_pts'] for r in added})
    assert all(exact[r['source_pts']]==r['source_frame_index'] for r in rows+added)
    assert all(Fraction(r['requested_fraction'])%1==Fraction(3,4) for r in added)
    assert sum(m['status']=='added' for report in reports for m in report['mappings'])==len(added)


def test_disabled_region_produces_no_assets(tmp_path):
    rows,scores,video=pool();scores=[.9 if i in (0,10) else .1 for i in range(40)]
    plan=build_regions(rows,scores,video,dict(CFG,seed_budget=2))
    added,assets,reports,_=decode_regions(video,rows,plan,CFG,tmp_path/'images')
    assert not added and not assets and all(not r['enabled'] for r in reports)


def test_pipeline_combines_new_and_old_without_new_itm(tmp_path):
    path=tmp_path/'pipeline.mp4';make_video(path,frames=180)
    class FakeScorer:
        """不加载模型，计数验证新帧只进入纯视觉接口。"""
        def __init__(self):self.coarse=0;self.fine=0
        def tokenize(self,question):return {'input_ids':np.array([[1,2,3]])}
        def score_batch_features(self,encoded,images):
            scores=[1-(self.coarse+i)/1000 for i in range(len(images))];self.coarse+=len(images)
            return scores,np.tile(np.array([1.,0.],np.float32),(len(images),1)),0.,0.
        def visual_batch(self,images):
            self.fine+=len(images)
            return np.tile(np.array([0.,1.],np.float32),(len(images),1)),0.,0.
    scorer=FakeScorer()
    selection,features,timing=prepare_density(path,'A question',scorer,CFG,tmp_path/'fine')
    assert scorer.coarse==selection['initial_count'] and scorer.fine==selection['new_candidate_count']>0
    assert features.shape[0]==len(selection['candidates'])
    assert len(selection['selected_frames'])==16
    assert any(i>=selection['initial_count'] for i in selection['selected_indices'])
    from videoqa_methods.density_run import verify_selection
    verify_selection(selection,features,CFG)


def test_incremental_selection_matches_independent_full_recompute():
    rng=np.random.default_rng(2027)
    for _ in range(20):
        rows,scores,video=pool(80);scores=rng.random(80).tolist()
        cfg=dict(CFG,window_radius_seconds='1')
        plan=build_regions(rows,scores,video,cfg)
        features=rng.normal(size=(80,8)).astype(np.float32);features/=np.linalg.norm(features,axis=1,keepdims=True)
        chosen,trace=select_joint(rows,features,plan,video,cfg)
        selected={r['region_id']:[r['anchor_index']] for r in plan['regions']}
        for recorded in trace['rounds']:
            proposals=[]
            for region in plan['regions']:
                rid=region['region_id'];ids=region['coarse_indices']
                redundant=np.maximum.reduce([np.clip(features[ids]@features[j],0,1) for j in selected[rid]])
                for pos,i in enumerate(ids):
                    if i not in selected[rid]:
                        proposals.append((region['weight']*(1-float(redundant[pos])),i,rid))
            _,i,rid=min(proposals,key=lambda x:(-x[0],rows[x[1]]['source_pts'],x[1]))
            assert recorded['winner']['candidate_index']==i
            selected[rid].append(i)
        assert chosen==sorted(i for values in selected.values() for i in values)


def test_recovery_rejects_unknown_or_uncertain_call(tmp_path):
    rows=[dict(question_id='q')]
    write_json(tmp_path/'calls/answer/q.json',dict(status='started_may_generate',protocol_sha256='fp',question_id='q'))
    with pytest.raises(ProtocolError):recovery_check(tmp_path,rows,'fp')


def test_completed_call_reused_without_generation(tmp_path):
    ledger=CallLedger(tmp_path,'q','fp','0');calls=[]
    def answer(state):
        """合成调用状态替代模型，只检查持久化账本行为。"""
        calls.append(1);state.update(started=True,returned=True);return {'raw_output':'A'}
    first=ledger.invoke('answer',answer)
    assert ledger.invoke('answer',answer)==first and calls==[1]
