"""固定10题、最多20次的协议桥接；先验证输入，生成结果不用于选择协议。"""
import copy
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,utc,exclusive,digest,invoke_once
from videoqa_full.prepare import environment
from .common import (SOURCE,LMMS,WFS,PROTOCOL,QUESTION_IDS,CONDITIONS,GENERATION,POST_PROMPT,
                     question_rows,manifest_rows,source_selection,framework_identity,load_task,task_doc)
from .frames import FrameProvider


def synthetic_row(n):
    """构造非数据集问题，合成生成不得混入真实问答或正确率。"""
    return dict(question_id=f'synthetic-{n}',video_id='synthetic',stratum='short',domain='Knowledge',
        task_type='Object Recognition',question='Which color is visible?',options=['A. Red.','B. Blue.','C. Green.','D. Gray.'],
        answer_index=1,original={'sub_category':'Humanity & History'})


def evaluate_once(backend,adapter,condition,frames,selection,row,state):
    """接口：只选择问答协议；两条件共享相同原始RGB，不在此重选源帧。"""
    from .llava_adapter import observe_generate
    video=selection['video'];times=[r['timestamp_seconds'] for r in selection['selected_frames']]
    if condition=='old':
        call=lambda:backend.answer(frames,times,video['duration_seconds'],row['question'],row['options'],generation_state=state)
        limit=8
    else:
        call=lambda:adapter.ask(frames,times,video['duration_seconds'],row,video['path'])
        limit=16
    answer,observed=observe_generate(backend,frames,limit,state,call)
    return dict(answer=answer,observed=observed)


def synthetic(run,gpu):
    """接口：只跑非数据集的1/11/14/16帧两协议合成问答，先于真实调用。"""
    import torch
    from PIL import Image
    from videoqa_runtime.llava_backend import LlavaBackend
    from videoqa_runtime.baseline_run import available_gpus
    from .llava_adapter import FrozenInputLlavaVid
    run=Path(run);assert gpu in available_gpus([gpu])[0]
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    status=dict(status='running',synthetic_generations=0,real_generations=0,checks=[],framework=framework_identity(),started_at=utc())
    try:
        backend=LlavaBackend();adapter=FrozenInputLlavaVid(backend);status['load_report']=backend.load_report
        for n in (1,11,14,16):
            frames=[Image.new('RGB',(720,480),(20,50+i*8,140)) for i in range(n)]
            selection=dict(video=dict(path='synthetic.mp4',duration_seconds=n+1),selected_frames=[dict(timestamp_seconds=float(i)) for i in range(n)])
            pair={}
            for condition in CONDITIONS:
                state={};status['synthetic_generations']+=1
                pair[condition]=evaluate_once(backend,adapter,condition,frames,selection,synthetic_row(n),state)
                assert state=={'started':True,'returned':True}
            assert pair['old']['observed']['pixel_sha256']==pair['reference']['observed']['pixel_sha256']
            status['checks'].append(dict(frame_count=n,results=pair))
            for im in frames:im.close()
        status['status']='passed'
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        status['finished_at']=utc();durable(run/'llava_synthetic_acceptance.json',status)


def prepare(run):
    """冻结桥接协议、来源、实际关键依赖和任务函数；不生成真实答案。"""
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    assert read_json(run/'offline_status.json')['status']=='completed','Finish offline source index first'
    # 用框架依赖的Decord独立核对源帧总数；选择位置仍按已冻结的固定帧分支。
    import decord
    import numpy as np
    native_check=run/'framework_uniform_check.json'
    uniform_summary=read_json(run/'uniform_summary.json')
    if native_check.exists():
        assert read_json(native_check)['uniform_summary_sha256']==sha256(run/'uniform_summary.json')
    else:
        checked=[]
        for vid,h in uniform_summary['record_hashes'].items():
            p=run/'uniform_records'/(vid+'.json');assert sha256(p)==h
            record=read_json(p);vr=decord.VideoReader(record['video']['path'],num_threads=2)
            assert len(vr)==record['source_frame_count'], 'Decoder frame count mismatch: '+vid
            ids=np.linspace(0,len(vr)-1,min(16,len(vr)),dtype=int).tolist()
            assert ids==[f['source_frame_index'] for f in record['new_frames']]
            checked.append(vid)
        assert len(checked)==900
        durable(native_check,dict(status='passed',videos=checked,count=900,
            uniform_summary_sha256=sha256(run/'uniform_summary.json'),decord_version=decord.__version__))
    frameworks=framework_identity();task=load_task();data=manifest_rows()
    for row in data:
        mapped=task_doc(row)
        assert all(mapped[k]==row['original'][k] for k in mapped)
    files=sorted((ROOT/'src/videoqa_lmms_bridge').glob('*.py'))
    files+=[ROOT/'scripts/run_lmms_bridge.py',ROOT/'configs/lmms_bridge_constraints.txt']
    code={str(p.relative_to(ROOT)):sha256(p) for p in files}
    src={};sources={}
    for row in question_rows():
        q=row['question_id'];selected,h=source_selection(q);sources[q]=dict(selection_sha256=h,source_path=selected['source_path'],source_sha256=selected['source_sha256'])
    for name in ('utils.py','videomme.yaml'):
        src[name]=sha256(WFS/name)
    from videoqa_full.prepare import verify_frozen_rd
    verify_frozen_rd()
    from importlib.metadata import version
    old=read_json(ROOT/'configs/runtime_environment.json')['packages']
    critical={n:version(n) for n in ('torch','torchvision','transformers','tokenizers','numpy','pillow','av','safetensors','accelerate','timm')}
    for n,v in critical.items():
        assert v.split('+')[0]==old[n].split('+')[0],(n,v,old[n])
    spec=dict(version=PROTOCOL,kind='fixed-input-diagnostic-not-algorithm-score',question_ids=list(QUESTION_IDS),
        conditions=list(CONDITIONS),maximum_real_calls=20,framework=frameworks,task_sources=src,
        code_sha256=code,source_selections=sources,critical_versions=critical,
        environment=environment('/App/conda/envs/explore_video_qa_lmms/bin/python'),
        old_generation_max_new_tokens=8,reference_generation=GENERATION,reference_post_prompt=POST_PROMPT,
        same_source_frames=True,same_model_object=True,model_template='qwen_1_5',dtype='bfloat16',attention='sdpa',
        reference_extra_time_instruction=False,old_extra_time_instruction=True,seed=2027,
        uniform_qa_in_bridge='historical uniform frames only; new native uniform is offline-only',
        native_uniform_check_sha256=sha256(native_check),
        new_full_evaluation_authorized=False)
    if (run/'bridge_protocol.json').exists():assert read_json(run/'bridge_protocol.json')==spec
    else:
        durable(run/'bridge_protocol.json',spec)
        for p in files:
            dest=run/'bridge_code_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        for name in src:
            dest=run/'task_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(WFS/name,dest)
    # 原始输出重解析，只读取已完成的8100条，不新增生成。
    source_csv=SOURCE/'llava_per_question.csv'
    import csv
    changes=[];count=0
    for r in csv.DictReader(source_csv.open()):
        parsed=task.extract_characters_regex(r['raw_output'])
        if parsed!=r['parsed_answer']:changes.append(dict(method=r['method'],question_id=r['question_id'],old=r['parsed_answer'],new=parsed))
        count+=1
    assert count==8100
    durable(run/'historical_parser_comparison.json',dict(n=count,changes=changes,real_calls=0,source_sha256=sha256(source_csv),task_sha256=src['utils.py']))
    return spec


def run_bridge(run,gpu):
    """后台诊断入口：必须先完成900视频离线与合成；调用开始失败也占20次额度。"""
    import torch
    from videoqa_runtime.llava_backend import LlavaBackend
    from videoqa_runtime.baseline_run import available_gpus
    from .llava_adapter import FrozenInputLlavaVid
    run=Path(run)
    with exclusive(run/'bridge.lock'):
        spec=prepare(run);fp=sha256(run/'bridge_protocol.json')
        assert read_json(run/'offline_status.json')['status']=='completed'
        assert read_json(run/'uniform_summary.json')['videos']==900
        assert read_json(run/'llava_synthetic_acceptance.json')['status']=='passed'
        allowed={(q,c) for q in QUESTION_IDS for c in CONDITIONS}
        calls=list((run/'diagnostic/calls').glob('*/*.json'));assert len(calls)<=20
        for p in calls:
            assert (p.stem,p.parent.name) in allowed
            r=read_json(p);assert r['protocol_sha256']==fp and r['status']=='completed','Uncertain call: '+str(p)
        existing=list((run/'bridge_results').glob('*.json'))
        assert all(p.stem in QUESTION_IDS for p in existing)
        if len(existing)==10:
            from .report import summarize
            summarize(run);return
        assert gpu in available_gpus([gpu])[0]
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        backend=LlavaBackend();adapter=FrozenInputLlavaVid(backend);t=time.perf_counter()
        status=dict(status='running',pid=os.getpid(),session_id=os.getsid(0),gpu=gpu,completed=len(existing),active=None,maximum_calls=20)
        def publish():durable(run/'bridge_status.json',dict(status,elapsed_seconds=time.perf_counter()-t,utc=utc()))
        publish()
        try:
            for ordinal,row in enumerate(question_rows()):
                q=row['question_id'];dest=run/'bridge_results'/(q+'.json')
                if dest.exists():continue
                status['active']=q;publish()
                saved,h=source_selection(q);assert h==spec['source_selections'][q]['selection_sha256']
                assert sha256(Path(saved['selection']['video']['path']))==saved['source_video_sha256']
                provider=FrameProvider(saved['selection']);frames,hashes=provider.decode()
                checkpoint=run/'bridge_inputs'/(q+'.json')
                inputs=dict(question_id=q,source_selection_sha256=h,selection=saved['selection'],rgb_sha256=hashes)
                if checkpoint.exists():assert read_json(checkpoint)==inputs
                else:durable(checkpoint,inputs)
                old_record=read_json(ROOT/saved['source_path']);assert sha256(ROOT/saved['source_path'])==saved['source_sha256']
                order=CONDITIONS if ordinal%2==0 else CONDITIONS[::-1];answers={}
                for condition in order:
                    answers[condition],reused,checkpoint_seconds=invoke_once(run/'diagnostic',q,condition,fp,gpu,
                        lambda state:evaluate_once(backend,adapter,condition,frames,saved['selection'],row,state))
                    assert len(list((run/'diagnostic/calls').glob('*/*.json')))<=20
                assert answers['old']['observed']['pixel_sha256']==answers['reference']['observed']['pixel_sha256']
                historical=old_record['answer']['parsed_answer'];old_answer=answers['old']['answer']['parsed_answer']
                record=dict(question_id=q,video_id=row['video_id'],stratum=row['stratum'],question=row['question'],options=row['options'],
                    reference_answer='ABCD'[row['answer_index']],historical_answer=historical,
                    repeat_agreement=historical==old_answer,attribution_status='eligible' if historical==old_answer else 'paused_repeatability',
                    conditions=answers,order=list(order),gpu=gpu,rgb_sha256=hashes,source_selection_sha256=h,
                    protocol_sha256=fp,completed_at=utc())
                durable(dest,record)
                for im in frames:im.close()
                status['completed']+=1;publish()
            status.update(status='completed',active=None);publish()
            from .report import summarize
            summarize(run)
        except BaseException as error:
            status.update(status='paused',error=repr(error),traceback=traceback.format_exc());publish();raise
