"""针对本次明确缺陷的小型回归测试，不加载权重或运行真实问答。"""
import inspect
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
from videoqa_runtime.common import ROOT,write_json,sha256
from videoqa_runtime.protocol import question_text
from videoqa_runtime.baseline_records import validate_result,check_recovery
from videoqa_runtime.baseline_run import preflight
from videoqa_runtime.llava_backend import LlavaBackend
sys.path.insert(0,str(ROOT/'scripts'))
import merge_full_completion as merger


def fixture(n,protocol='videoqa-llava-16-cap-v1'):
    """构造完整的合成输入记录，只用于CPU侧校验。"""
    row=dict(question_id='q',video_id='v',question='Question?',options=['a','b','c','d'],answer_index=0)
    frames=[dict(candidate_index=i,source_pts=i,timestamp_seconds=float(i)) for i in range(n)]
    answer=dict(protocol=protocol,pixel_shape=[n,3,384,384],visual_tokens=n*210,prefill_tokens=19+n*210,
                text_input_tokens=20,raw_output='A.',parsed_answer='A',
                prompt=question_text(row['question'],row['options'],n+1,list(range(n))))
    record=dict(status='completed',protocol_sha256='p',question_id='q',video_id='v',question=row['question'],options=row['options'],
                method='uniform',selection=dict(candidates=frames,selected_frames=frames,scores=[],video=dict(duration_seconds=n+1)),
                answer=answer,reference_answer='A',correct=True,application_cache_hits=0,generation_attempts=1,
                timings=dict(stage_sum_seconds=1.,unattributed_seconds=0.,end_to_end_seconds=1.,blip_forward_seconds=0.))
    return row,record


def test_old_preflight_call_still_binds():
    """旧方法入口只传rows仍合法，不触发任何资产扫描或模型加载。"""
    inspect.signature(preflight).bind([])


def test_frame_timestamp_mismatch_rejected_before_backend_access():
    """没有初始化模型的对象也能在入口拒绝错配，证明保护早于生成。"""
    backend=LlavaBackend.__new__(LlavaBackend)
    with pytest.raises(ValueError,match='equal length'):
        backend.answer([object()]*14,list(range(11)),20,'q',['a','b','c','d'])


@pytest.mark.parametrize('n',[1,11,14,16])
def test_cap_validates_actual_frames(n):
    row,record=fixture(n)
    assert validate_result(record,row,'p',dict(frame_budget=16))


def test_fixed_protocol_cannot_accept_short_or_cap_records():
    row,record=fixture(14,'videoqa-llava-16-v1')
    with pytest.raises(ValueError,match='requires 16'):validate_result(record,row,'p',dict(frames=16))
    row,record=fixture(16)
    with pytest.raises(ValueError,match='differs'):validate_result(record,row,'p',dict(frames=16))
    record['answer']['protocol']='videoqa-llava-16-v1'
    assert validate_result(record,row,'p',dict(frames=16))


def test_saved_result_is_counted_after_pause(tmp_path):
    """暂停收尾重扫时计入在途落盘，未完成且无结果的生成仍被阻止。"""
    row,record=fixture(14)
    write_json(tmp_path/'protocol.json',dict(frame_budget=16))
    write_json(tmp_path/'results/uniform/q.json',record)
    write_json(tmp_path/'attempts/q/uniform.01.json',dict(question_id='q',method='uniform',status='running_may_generate'))
    assert check_recovery(tmp_path,[row],'p')=={('q','uniform')}
    write_json(tmp_path/'attempts/q/topk.01.json',dict(question_id='q',method='topk',status='running_may_generate'))
    with pytest.raises(ValueError,match='Uncertain'):check_recovery(tmp_path,[row],'p')


def test_merge_rejects_manifest_or_snapshot_tampering(tmp_path,monkeypatch):
    """在读取结果前校验清单和源码；所有写入均为隔离测试夹具。"""
    monkeypatch.setattr(merger,'ROOT',tmp_path)
    manifest=tmp_path/'manifest.json';run=tmp_path/'run'
    write_json(manifest,dict(rows=[]))
    spec=dict(dataset_manifest='manifest.json',dataset_manifest_sha256='0'*64,code_sha256={})
    write_json(run/'protocol.json',spec)
    with pytest.raises(ValueError,match='manifest hash'):merger.load_run(run,manifest,'test')
    spec.update(dataset_manifest_sha256=sha256(manifest),code_sha256={'entry.py':'0'*64})
    (run/'code_snapshot').mkdir();(run/'code_snapshot/entry.py').write_text('pass\n')
    write_json(run/'protocol.json',spec)
    with pytest.raises(ValueError,match='snapshot changed'):merger.load_run(run,manifest,'test')


def extraction_script(tmp_path):
    """把脚本的固定数据根替换为临时夹具，不接触真实视频目录。"""
    text=(ROOT/'scripts/extract_video_mme_full.sh').read_text().replace('/home/admin/projects/explore_video_qa',str(tmp_path))
    path=tmp_path/'extract.sh';path.write_text(text)
    return path


def test_extraction_missing_manifest_fails_before_writes(tmp_path):
    result=subprocess.run(['bash',str(extraction_script(tmp_path))],capture_output=True,text=True)
    assert result.returncode!=0 and 'Missing or empty manifest' in result.stderr
    assert not (tmp_path/'data/videos').exists() and 'EXTRACT DONE' not in result.stdout


def test_extraction_subtitle_failure_never_claims_success(tmp_path):
    """仅解压20个几字节的合成归档；损坏字幕应中止，不打印完成。"""
    archives=tmp_path/'data/archives/video_mme_full';archives.mkdir(parents=True)
    rows=[]
    for i in range(20):
        p=archives/f'videos_chunked_{i:02}.zip'
        with zipfile.ZipFile(p,'w') as z:z.writestr(f'data/{i}.mp4',b'fixture')
        rows.append(f'{p.name}\t{p.stat().st_size}')
    subtitle=archives/'subtitle.zip';subtitle.write_bytes(b'not a zip')
    rows.append(f'subtitle.zip\t{subtitle.stat().st_size}')
    manifest=tmp_path/'data/manifests/video_mme_full_filelist.txt';manifest.parent.mkdir();manifest.write_text('\n'.join(rows)+'\n')
    result=subprocess.run(['bash',str(extraction_script(tmp_path))],capture_output=True,text=True)
    assert result.returncode!=0 and 'EXTRACT DONE' not in result.stdout
