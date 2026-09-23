"""源帧层离线核对：全片真实PTS建立索引，不按平均FPS推断帧时间。"""
import concurrent.futures
import os
import time
import subprocess
import importlib.metadata
from pathlib import Path
from fractions import Fraction
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,utc,exclusive,save_array,validate_frames
from .common import SOURCE,LMMS,LMMS_COMMIT,manifest_rows,source_selection


def native_indices(count,budget=16):
    """参考加载器的固定帧分支：源编号linspace取整；短源不重复补齐。"""
    if count<1 or budget<1:raise ValueError('Empty video or invalid budget')
    return np.linspace(0,count-1,min(count,budget),dtype=int).tolist()


def rows_at_indices(pts,video,indices):
    """接口：源帧号映射到真实PTS，不把展示秒数或平均帧率当作身份。"""
    out=[];tb=Fraction(video['time_base']);origin=video['start_pts']
    for ordinal,i in enumerate(indices):
        instant=(int(pts[i])-origin)*tb
        out.append(dict(candidate_index=ordinal,source_frame_index=i,source_pts=int(pts[i]),
            source_seconds=float(int(pts[i])*tb),timestamp_seconds=float(instant),requested_seconds=float(instant)))
    validate_frames(dict(video=video,selected_frames=out))
    return out


def inspect_video(job):
    """单视频完整解码只收集PTS；最多保留整数索引，不把全片RGB留在内存。"""
    import av
    run=Path(job['run']);old=job['old'];video_id=old['video_id'];dest=run/'uniform_records'/(video_id+'.json')
    path=Path(old['selection']['video']['path'])
    if dest.exists():
        saved=read_json(dest)
        assert saved['source_video_sha256']==old['source_video_sha256']
        assert saved['old_selection_hash']==job['old_hash']
        assert sha256(run/saved['pts_file'])==saved['pts_sha256']
        assert sha256(path)==saved['source_video_sha256']
        return saved
    start=time.perf_counter()
    # 步骤1：全解码的呈现顺序就是源帧编号，兼容B帧与变帧率。
    assert sha256(path)==old['source_video_sha256']
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=2
        tb=Fraction(stream.time_base);origin=stream.start_time or 0
        assert stream.duration is not None and stream.duration>0
        pts=np.asarray([frame.pts for frame in container.decode(stream)],dtype=np.int64)
        assert len(pts)>0 and np.all(np.diff(pts)>0)
        video=dict(path=str(path),start_pts=origin,time_base=str(tb),duration_seconds=float(stream.duration*tb),
                   duration_fraction=str(stream.duration*tb))
    # 步骤2：旧16帧与独立源索引核对，新16帧只按参考规则产生。
    old_frames=old['selection']['selected_frames']
    for f in old_frames:assert int(pts[f['source_frame_index']])==f['source_pts']
    indices=native_indices(len(pts));new=rows_at_indices(pts,video,indices)
    old_ids={f['source_frame_index'] for f in old_frames};new_ids=set(indices)
    time_old=[f['timestamp_seconds'] for f in old_frames];time_new=[f['timestamp_seconds'] for f in new]
    n=min(len(time_old),len(time_new))
    file=run/'pts'/(video_id+'.npy');h=save_array(file,pts)
    # 步骤3：只称时间覆盖变化，不把新位置直接标成证据改善。
    result=dict(video_id=video_id,question_ids=job['question_ids'],source_video_sha256=old['source_video_sha256'],
        old_selection_hash=job['old_hash'],pts_file=str(file.relative_to(run)),pts_sha256=h,
        video=video,source_frame_count=len(pts),old_frames=old_frames,new_frames=new,
        old_count=len(old_frames),new_count=len(new),overlap=len(old_ids&new_ids),
        identical=[f['source_frame_index'] for f in old_frames]==indices,
        pts_step_count=len(np.unique(np.diff(pts))),
        variable_pts_steps=bool(len(np.unique(np.diff(pts)))>1),
        paired_position_abs_seconds=[abs(time_old[i]-time_new[i]) for i in range(n)] if len(time_old)==len(time_new) else None,
        first_shift_seconds=time_new[0]-time_old[0],last_shift_seconds=time_new[-1]-time_old[-1],
        old_span_seconds=time_old[-1]-time_old[0],new_span_seconds=time_new[-1]-time_new[0],
        decode_seconds=time.perf_counter()-start,status='completed',new_model_calls=0)
    durable(dest,result)
    return result


def offline(run,workers=8):
    """接口：900视频对照独立后台运行；异常暂停，完整且指纹一致的条目可复用。"""
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'offline.lock'):
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=LMMS,text=True).strip()==LMMS_COMMIT
        code_paths=['src/videoqa_lmms_bridge/frames.py','src/videoqa_lmms_bridge/common.py','scripts/analyze_lmms_uniform.py']
        identity=dict(version='lmms-native-uniform-pts-v1',framework_commit=LMMS_COMMIT,
            loader_sha256=sha256(LMMS/'lmms_eval/models/simple/llava_vid.py'),
            source_protocol_sha256=sha256(SOURCE/'protocol.json'),
            code_sha256={p:sha256(ROOT/p) for p in code_paths},
            packages={p:importlib.metadata.version(p) for p in ('av','numpy')},budget=16,
            rule='linspace(0,N_source-1,min(16,N_source),dtype=int); source timestamps from full decode')
        if (run/'uniform_protocol.json').exists():assert read_json(run/'uniform_protocol.json')==identity
        else:
            durable(run/'uniform_protocol.json',identity)
            for name in code_paths:
                import shutil
                dest=run/'offline_code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/name,dest)
        jobs={};spec=read_json(SOURCE/'protocol.json');start=time.perf_counter()
        for row in manifest_rows():
            qid=row['question_id'];video_id=row['video_id'];old,h=source_selection(qid)
            if video_id in jobs:
                assert old['selection']==jobs[video_id]['old']['selection']
                jobs[video_id]['question_ids'].append(qid)
            else:jobs[video_id]=dict(run=str(run),old=old,old_hash=h,question_ids=[qid])
        assert len(jobs)==900
        status=dict(status='running',pid=os.getpid(),session_id=os.getsid(0),completed=0,total=900,new_model_calls=0)
        def publish():
            durable(run/'offline_status.json',dict(status,elapsed_seconds=time.perf_counter()-start,utc=utc()))
        publish();results=[]
        ordered=sorted(jobs.values(),key=lambda j:-j['old']['selection']['video']['duration_seconds'])
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures=[executor.submit(inspect_video,j) for j in ordered]
                for future in concurrent.futures.as_completed(futures):
                    result=future.result();results.append(result);status['completed']=len(results);publish()
                    print('Uniform index',len(results),'/900',flush=True)
            assert len(results)==900
            summary=dict(videos=900,questions=2700,identical_videos=sum(r['identical'] for r in results),
                mean_overlap=float(np.mean([r['overlap'] for r in results])),
                overlap_histogram={str(n):sum(r['overlap']==n for r in results) for n in range(17)},
                variable_pts_step_videos=[r['video_id'] for r in results if r['variable_pts_steps']],
                short_pool_cases=[{k:r[k] for k in ('video_id','question_ids','old_count','new_count')} for r in results if r['old_count']<16],
                record_hashes={r['video_id']:sha256(run/'uniform_records'/(r['video_id']+'.json')) for r in results},
                elapsed_seconds=time.perf_counter()-start,new_model_calls=0,
                limitation='variable PTS increments can also reflect time-base rounding; not automatically a semantic VFR label')
            durable(run/'uniform_summary.json',summary);status['status']='completed';publish()
        except BaseException as exc:
            status.update(status='paused',error=repr(exc));publish();raise


class FrameProvider:
    """接口：只按源选择记录或video_id返回帧，不接收原题答案/题型/模型预测。"""
    def __init__(self,selection):
        validate_frames(selection);self.selection=selection

    @classmethod
    def from_native(cls,run,video_id):
        record=read_json(Path(run)/'uniform_records'/(video_id+'.json'))
        return cls(dict(video=record['video'],selected_frames=record['new_frames']))

    def decode(self):
        """按精确PTS只解码一次；两桥接入口复用同一RGB对象与像素指纹。"""
        import hashlib
        from videoqa_methods.region_context import read_exact_frames
        frames=read_exact_frames(self.selection['video'],self.selection['selected_frames'])
        hashes=[hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in frames]
        return frames,hashes
