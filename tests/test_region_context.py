"""R的保护、区域连接、补位和解码测试；不运行任何真实模型。"""
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
from videoqa_methods.region_context import make_replacement_plan,select_from_neighborhoods,read_neighborhoods_and_select,CONFIG
from videoqa_methods.region_run import check_gate
from videoqa_methods.followups import CallLedger
from videoqa_runtime.common import write_json,sha256
from videoqa_runtime.baseline_selection import ProtocolError,iter_candidate_frames


def pool(n=40):
    """构造真实时间/目标网格分离的候选，top16默认为连续前16帧。"""
    rows=[dict(candidate_index=i,source_pts=4*i+1,source_frame_index=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(n)]
    scores=[1-i/(n+1) for i in range(n)]
    video=dict(time_base='1/4',start_pts=0,duration_seconds=float(n))
    return rows,scores,video


def added(rows,pts):
    """新增候选延续编号，保存精确PTS供距离比较。"""
    return [dict(candidate_index=len(rows)+j,source_pts=p,source_frame_index=p,timestamp_seconds=p/4,requested_seconds=p/4) for j,p in enumerate(pts)]


def test_protection_ties_and_low_score_removal():
    c,s,v=pool();s[:16]=[.5]*16
    p=make_replacement_plan(c,s,v)
    # 使池外分数低于Top16，平分时峰值为最早源帧。
    s[16:]=[0.]*(len(s)-16);p=make_replacement_plan(c,s,v)
    assert p['regions'][0]['peak_index']==0
    assert p['protected_indices']==[0,15]
    assert p['potential_removals']==[14,13,12,11]


def test_peak_and_both_endpoints_are_protected():
    c,s,v=pool();s=[.1]*40;s[:16]=[.5]*16;s[8]=.9
    p=make_replacement_plan(c,s,v)
    assert set(p['protected_indices'])=={0,8,15}
    assert not set(p['potential_removals'])&{0,8,15}


def test_connection_uses_requested_grid_not_source_rounding():
    c,s,v=pool();c[1]['source_pts']+=1
    p=make_replacement_plan(c,s,v)
    assert p['regions'][0]['indices'][:2]==[0,1]


@pytest.mark.parametrize('n',[1,11,14,15])
def test_short_pool_no_refinement(n):
    c,s,v=pool(n);p=make_replacement_plan(c,s,v)
    assert p['short_pool'] and not p['windows'] and not p['potential_removals']
    assert select_from_neighborhoods(c,s,v,p,[])['selected_indices']==list(range(n))


def test_distributed_topk_keeps_original():
    c,s,v=pool();s=[.9 if i%2==0 and i<32 else .1 for i in range(40)]
    p=make_replacement_plan(c,s,v)
    assert len(p['regions'])==16 and not p['windows']


def test_clipping_and_nearest_boundary_selection():
    c,s,v=pool();p=make_replacement_plan(c,s,v)
    assert p['windows'][0]['left_fraction']=='0'
    new=added(c,[0,62,63,64])
    r=select_from_neighborhoods(c,s,v,p,new)
    assert len(r['selected_frames'])==16 and len(r['replacements'])==4
    assert r['replacements'][0]['added_index']==40  # 前0秒与后15.5秒同距，前者PTS更早。
    assert set(p['protected_indices'])<=set(r['selected_indices'])


def test_no_neighbor_keeps_old_and_scores_do_not_rank_neighbors():
    c,s,v=pool(16);p=make_replacement_plan(c,s,v)
    assert select_from_neighborhoods(c,s,v,p,[])['selected_indices']==p['topk_indices']
    # 旧低分候选也可补位，排序只取决于距边界距离。
    c,s,v=pool();p=make_replacement_plan(c,s,v)
    r=select_from_neighborhoods(c,s,v,p,[])
    assert r['replacements'][0]['added_index']==16


def test_duplicate_new_source_is_rejected():
    c,s,v=pool();p=make_replacement_plan(c,s,v)
    with pytest.raises(ProtocolError):select_from_neighborhoods(c,s,v,p,added(c,[c[0]['source_pts']]))


def test_region_quota_does_not_borrow_slots():
    c,s,v=pool(80);s=[0.]*80
    top=list(range(5,9))+list(range(20,32))
    for rank,i in enumerate(top):s[i]=1-rank*.01
    p=make_replacement_plan(c,s,v);r=select_from_neighborhoods(c,s,v,p,added(c,[19,34,79,128]))
    for region in p['regions']:
        assert sum(x['region_id']==region['region_id'] for x in r['replacements'])<=len(region.get('replacement_slots',[]))


def test_vfr_refinement_source_indices_match_full_decode(tmp_path):
    import av
    path=tmp_path/'vfr.mp4'
    with av.open(str(path),'w') as out:
        stream=out.add_stream('mpeg4',rate=8);stream.width=64;stream.height=48;stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,8)
        for i in range(180):
            f=av.VideoFrame.from_ndarray(np.full((48,64,3),i%255,np.uint8),format='rgb24');f.pts=i+(i//17);f.time_base=Fraction(1,8)
            for pkt in stream.encode(f):out.mux(pkt)
        for pkt in stream.encode():out.mux(pkt)
    video={};c=[r for r,f in iter_candidate_frames(path,video)];scores=[1-i/(len(c)+1) for i in range(len(c))]
    result,_=read_neighborhoods_and_select(dict(video=video,candidates=c,scores=scores))
    with av.open(str(path)) as inp:indices={f.pts:i for i,f in enumerate(inp.decode(video=0))}
    assert all(indices[r['source_pts']]==r['source_frame_index'] for r in result['candidates'])
    assert result['new_candidate_count']<=24 and all(t['retained']<=3 for t in result['fine_decode_trace'])


def test_failed_gate_cannot_start_formal(tmp_path):
    write_json(tmp_path/'offline_manifest.json',dict(result_sha256={}))
    write_json(tmp_path/'gate.json',dict(status='failed'))
    with pytest.raises(ProtocolError):check_gate(tmp_path,'fp')
    assert not (tmp_path/'calls').exists()


def test_completed_answers_reused_uncertain_never_repeated(tmp_path):
    ledger=CallLedger(tmp_path,'q','fp','0');counter=[]
    def answer(state):
        """记录假调用次数，证明恢复不会再次执行。"""
        counter.append(1);state.update(started=True,returned=True);return dict(raw_output='A')
    ledger.invoke('answer',answer);ledger.invoke('answer',answer);assert len(counter)==1
    write_json(tmp_path/'calls/answer/q.json',dict(status='started_may_generate',protocol_sha256='fp'))
    with pytest.raises(ProtocolError):ledger.invoke('answer',answer)
