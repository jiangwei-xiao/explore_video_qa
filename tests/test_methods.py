import ast
import copy
import json
import os
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import pywt
from scipy.signal import find_peaks

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_runtime.baseline_selection import iter_candidate_frames, select_topk, ProtocolError
from videoqa_methods.algorithm import segment_scores, compete, hotspot_candidates, reselect, membership
from videoqa_methods.refinement import decode_hotspots, fine_targets
from videoqa_methods.model_ops import parse_scope, ScopeClassifier, FeatureScorer, SCOPE_PROMPT
from videoqa_methods.pipeline import select_method, select_diagnostic
from videoqa_methods.records import check_recovery
from videoqa_methods.worker import order_for

CFG = read_json(ROOT / 'configs/method_v1.json')


def rows(n):
    return [dict(candidate_index=i, source_pts=4*i+1, source_frame_index=4*i+1,
                 source_seconds=i+.25, timestamp_seconds=i+.25, requested_seconds=i+.25) for i in range(n)]


def video(n):
    return dict(time_base='1/4', start_pts=0, duration_seconds=float(n))


def sections(candidates, scores, edges):
    return [dict(segment_id=j, start=a, end=b, start_pts=0 if a==0 else candidates[a]['source_pts'],
                 representative_score=float(max(scores[a:b])), start_seconds=float(a), end_seconds=float(b))
            for j,(a,b) in enumerate(zip(edges,edges[1:]))]


@pytest.mark.parametrize('n,kind', [(2,'zero'),(31,'zero'),(64,'constant'),(100,'peak'),(513,'noise')])
def test_segmentation_matches_frozen_official_logic(n,kind):
    path = ROOT / 'third_party/WFS-SB-reference/core.py'
    assert sha256(path) == CFG['wfs_reference_sha256']
    names = {'WFSEventDetector','compute_dwt_level','compute_min_peak_distance'}
    nodes = [x for x in ast.parse(path.read_text()).body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in names]
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    namespace = dict(np=np,pywt=pywt,find_peaks=find_peaks)
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
    x = np.zeros(n) if kind=='zero' else np.ones(n)*.4 if kind=='constant' else np.random.default_rng(2027).random(n)
    if kind=='peak': x[n//2:n//2+5]=1
    segments, report = segment_scores(x,rows(n),video(n),CFG)
    detector = namespace['WFSEventDetector']()
    level = namespace['compute_dwt_level'](n)
    detail = detector.reconstruct_detail(detector.decompose(x,level),n)
    peaks = detector.detect_peaks(detail,namespace['compute_min_peak_distance'](n))
    assert report['level']==level and report['peaks']==peaks.tolist()
    np.testing.assert_allclose(report['detail_signal'],detail,atol=0,rtol=0)
    assert [(s['start'],s['end']) for s in segments] == detector.create_segments(peaks,n)


def test_first_pick_ties_and_negative_gains_fill_budget():
    c=rows(20); r=[0.0]*20; f=np.tile(np.array([[1.,0.]],np.float32),(20,1))
    selected,quota,trace=compete(c,r,f,sections(c,r,[0,20]),.8,CFG)
    assert trace[0]['candidate_index']==0 and len(selected)==16 and quota==[16]
    assert all(step['gain']<0 for step in trace[1:])
    r=[.2]*20; r[17]=.9
    _,_,trace=compete(c,r,f,sections(c,r,[0,10,20]),.2,CFG)
    assert trace[0]['candidate_index']==17


def test_redundancy_is_segment_local():
    c=rows(4); r=[.9,.8,.7,.6]; f=np.tile(np.array([[1.,0.]],np.float32),(4,1))
    _,_,trace=compete(c,r,f,sections(c,r,[0,2,4]),.5,CFG,count=3)
    for step in trace:
        if step['segment_count_before']==0: assert step['duplicate']==0


def test_insufficient_or_invalid_features_stop():
    c=rows(4); r=[.5]*4; f=np.eye(4,dtype=np.float32)
    with pytest.raises(ProtocolError): compete(c,r,f,sections(c,r,[0,4]),.5,CFG)
    with pytest.raises(ProtocolError): compete(c,r,f*0,sections(c,r,[0,4]),.5,CFG,count=2)


def test_static_video_skips_refinement():
    c=rows(20); r=[.7]*20; f=np.tile(np.array([[1.,0.]],np.float32),(20,1)); seg=sections(c,r,[0,20])
    selected,q,_=compete(c,r,f,seg,.5,CFG)
    trace,hot=hotspot_candidates(c,r,f,selected,seg,q,video(20),CFG)
    assert not hot and trace[0]['reason']=='already_covered'
    chosen,_=reselect(c,r,f,seg,q,selected,len(c),.5,CFG)
    assert chosen==selected


def test_hotspot_touching_endpoints_overlap_and_limit():
    c=rows(20); r=[.5]*20; f=np.eye(20,dtype=np.float32)
    seg=sections(c,r,[0,4,8,12,16,20]); selected=[2,6,10,14,18]
    trace,hot=hotspot_candidates(c,r,f,selected,seg,[1]*5,video(20),CFG)
    assert len(hot)==2
    assert any(h['reason']=='overlap' for h in trace)
    for i,a in enumerate(hot):
        for b in hot[i+1:]: assert Fraction(a['right_fraction'])<Fraction(b['left_fraction']) or Fraction(b['right_fraction'])<Fraction(a['left_fraction'])


def test_fine_quota_is_fixed_even_when_zero_quota_region_has_best_new_frame():
    c=rows(20); r=[.6]*20; f=np.eye(22,dtype=np.float32)[:20]; seg=sections(c,r,[0,10,20])
    coarse=[0,1,2,3]
    c += [dict(candidate_index=20,source_pts=2,timestamp_seconds=.5),dict(candidate_index=21,source_pts=50,timestamp_seconds=12.5)]
    r += [.9,1.0]; f=np.eye(22,dtype=np.float32)
    chosen,_=reselect(c,r,f,seg,[4,0],coarse,20,.5,CFG)
    assert 20 in chosen and 21 not in chosen
    assert np.bincount(membership(c,seg)[chosen],minlength=2).tolist()==[4,0]


@pytest.mark.parametrize('text,label,fallback',[('local','LOCAL',False),(' GLOBAL\n','GLOBAL',False),('MIXED','MIXED',False),('LOCAL.','MIXED',True),('','MIXED',True)])
def test_scope_parser(text,label,fallback):
    assert parse_scope(text)==(label,fallback)


def test_scope_is_text_only_and_does_not_mutate_generation_settings(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    class Batch(dict):
        def to(self,device): return self
    class Tokenizer:
        eos_token_id=2; pad_token_id=0
        def __call__(self,text,**kwargs):
            self.prompt=text
            assert kwargs['truncation'] is False
            return Batch(input_ids=torch.ones((1,8),dtype=torch.long),attention_mask=torch.ones((1,8),dtype=torch.long))
        def batch_decode(self,x,**kwargs): return ['LOCAL']
    class Model:
        generation_config=SimpleNamespace(bos_token_id=1,max_new_tokens=99)
        def generate(self,ids,**kwargs):
            assert 'images' not in kwargs
            assert kwargs['generation_config'].max_new_tokens==8
            return torch.tensor([[3]])
    b=SimpleNamespace(tokenizer=Tokenizer(),model=Model(),context=32768)
    before=copy.deepcopy(vars(b.model.generation_config))
    state={}; result=ScopeClassifier(b).classify('What is visible?',state)
    assert result['label']=='LOCAL' and state['returned']
    assert '<image>' not in result['prompt'] and vars(b.model.generation_config)==before


def make_video(path,vfr=False):
    import av
    times=[]
    with av.open(str(path),'w') as output:
        stream=output.add_stream('mpeg4',rate=25 if vfr else 4)
        stream.width,stream.height,stream.pix_fmt=64,48,'yuv420p'
        if vfr:
            stream.time_base=Fraction(1,1000); stream.codec_context.time_base=Fraction(1,1000)
        instant=0
        for i in range(420 if vfr else 80):
            frame=av.VideoFrame.from_ndarray(np.full((48,64,3),i%230,np.uint8),format='rgb24')
            if vfr:
                frame.pts=instant; frame.time_base=Fraction(1,1000); instant += [25,45,80,50][i%4]
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)


@pytest.mark.parametrize('vfr',[False,True])
def test_refinement_source_indices_and_grid_are_exact(tmp_path,vfr):
    import av
    p=tmp_path/'video.mp4'; make_video(p,vfr)
    meta={}; c=[row for row,_ in iter_candidate_frames(p,meta)]
    tb=Fraction(meta['time_base']); duration=Fraction(str(meta['duration_seconds']))
    windows=[]
    for i in [0,7]:
        center=(c[i]['source_pts']-meta['start_pts'])*tb
        windows.append(dict(left_fraction=str(max(0,center-2)),right_fraction=str(min(duration,center+2)),center_fraction=str(center)))
    added,images,reports,times=decode_hotspots(meta,c,windows,CFG)
    with av.open(str(p)) as container:
        lookup={f.pts:i for i,f in enumerate(container.decode(video=0))}
    assert len(added)==len(images)<=24 and all(r['retained']<=12 for r in reports)
    assert not {r['source_pts'] for r in added}&{r['source_pts'] for r in c}
    assert all(lookup[r['source_pts']]==r['source_frame_index'] for r in added)
    assert all(Fraction(r['requested_fraction'])*4%4!=1 for r in added)


def test_integrated_mixed_and_no_refinement_equivalences(tmp_path):
    p=tmp_path/'video.mp4'; make_video(p)
    class Scorer:
        def tokenize(self,q): return {'input_ids':np.zeros((1,10))}
        def score_batch_features(self,encoded,images):
            return [float(np.asarray(im).mean())/255 for im in images],np.tile(np.array([[1.,0.]],np.float32),(len(images),1)),0.,0.
    class Classifier:
        def classify(self,q,state): return dict(label='MIXED',fallback=False,seconds=0.)
    results={v:select_method(p,'Question?',v,Scorer(),Classifier(),CFG,{})[1] for v in ['A','B','C']}
    assert results['A']['selected_indices']==results['B']['selected_indices']==results['C']['selected_indices']
    _,diagnostic,_=select_diagnostic({'pool':results['C']})
    assert diagnostic['selected_frames']==select_topk(results['A']['candidates'],results['A']['scores'])


def test_order_keeps_d_after_c():
    for i in range(6):
        order=order_for(i)
        assert set(order)==set('ABCD') and order.index('D')==order.index('C')+1


def test_closed_fine_window_is_capped_after_deduplication(tmp_path):
    # 中文说明：闭区间可能含13个非粗网格点，实际新增仍必须限制到12个。
    p=tmp_path/'video.mp4'; make_video(p)
    meta={}; c=[r for r,_ in iter_candidate_frames(p,meta)]
    h=dict(left_fraction='9/2',right_fraction='17/2',center_fraction='13/2')
    assert len(fine_targets(h,meta))==13
    added,_,records,_=decode_hotspots(meta,c,[h],CFG)
    assert len(added)==12 and any(r['reason']=='per_hotspot_cap' for r in records[0]['dropped'])


def test_method_interfaces_have_chinese_explanations():
    # 中文说明：后续维护时，接口定义必须保留可读的中文职责说明。
    import re
    for path in (ROOT/'src/videoqa_methods').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.FunctionDef):
                assert re.search(r'[\u4e00-\u9fff]',ast.get_docstring(node) or ''),(path.name,node.name)


def test_completed_scope_is_not_repeated_on_resume(tmp_path):
    p=tmp_path/'attempts/q'; p.mkdir(parents=True)
    (p/'B.01.json').write_text(json.dumps(dict(question_id='q',variant='B',status='failed',scope_state={'returned':True})))
    with pytest.raises(ProtocolError,match='classification'):
        check_recovery(tmp_path,[],'hash')


@pytest.mark.skipif(os.environ.get('VIDEOQA_GPU_TESTS')!='1',reason='Opt-in GPU-only feature validation')
def test_blip_scores_match_and_visual_cls_ignores_question():
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    scorer=FeatureScorer('/home/models/Salesforce/blip-itm-base-coco')
    images=[Image.new('RGB',(1280,720),'red'),Image.new('RGB',(1280,720),'blue')]
    first=scorer.tokenize('What color is visible?'); second=scorer.tokenize('How many animals are present?')
    baseline,_,_=scorer.score_batch(first,images)
    scores,features,_,_=scorer.score_batch_features(first,images)
    _,other,_,_=scorer.score_batch_features(second,images)
    np.testing.assert_allclose(baseline,scores,atol=1e-6,rtol=0)
    np.testing.assert_allclose(features,other,atol=1e-6,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(features,axis=1),1,atol=1e-6)
