"""冻结协议外的JSON身份表示修复：不修改语义或放宽像素/Token校验。"""
import json
from videoqa_full.state import digest
from videoqa_audit.region_scope_runtime import source_order
from videoqa_audit.scorer_comparison_recovery import canonical_source_order


def child_canonical(root,queue):
    """接口：spawn不继承父内存，子进程必须自己安装同一执行修复。"""
    from videoqa_audit.scorer_comparison_recovery import install_recovery
    from videoqa_scorer_comparison import runtime
    install_recovery(root)
    s=dict(video=dict(path='x',start_pts=0,time_base='1/30',duration_fraction='10'),
        selected_frames=[dict(source_frame_index=3,source_pts=3)])
    queue.put(runtime.source_order(s))


def test_source_identity_json_roundtrip_is_exact():
    """接口：同帧/同顺序落盘后仍能直接比较，摘要与修复前完全一致。"""
    selection=dict(video=dict(path='x.mp4',start_pts=0,time_base='1/30',duration_fraction='10'),
        selected_frames=[dict(source_frame_index=3,source_pts=3),dict(source_frame_index=9,source_pts=9)])
    original=source_order(selection);saved=json.loads(json.dumps(original))
    assert original!=saved
    assert canonical_source_order(selection)==saved
    assert digest(original)==digest(canonical_source_order(selection))


def test_source_change_not_hidden_by_canonicalization():
    """接口：只改容器类型，不允许编号/PTS/时钟/顺序变化被同一摘要掩盖。"""
    a=dict(video=dict(path='x.mp4',start_pts=0,time_base='1/30',duration_fraction='10'),
        selected_frames=[dict(source_frame_index=3,source_pts=3)])
    b=dict(a,selected_frames=[dict(source_frame_index=4,source_pts=4)])
    assert digest(canonical_source_order(a))!=digest(canonical_source_order(b))


def test_spawn_process_installs_canonical_representation(tmp_path):
    """接口：真实spawn仅作CPU身份检查，模型前向和生成均为0。"""
    import multiprocessing as mp
    from videoqa_full.state import durable
    durable(tmp_path/'preflight.json',{'test_only':True})
    ctx=mp.get_context('spawn');q=ctx.Queue();p=ctx.Process(target=child_canonical,args=(str(tmp_path),q))
    p.start();value=q.get(timeout=15);p.join(15)
    assert p.exitcode==0 and value['frames']==[[3,3]]
