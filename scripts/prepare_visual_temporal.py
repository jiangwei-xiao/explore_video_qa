"""原视觉预算下的联合重复度单版本准备，所有选帧/复核均无模型前向。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import argparse
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_methods.visual_temporal_redundancy import select_visual_temporal_redundancy
from videoqa_audit.coverage_objective_review import selected_indices
from videoqa_budget32.frames import FrameProvider,validate_frames
from analyze_coverage_objectives import BASE,DEV,FACTS
from prepare_temporal_coverage import render

METHOD='RD-1.2-exp26'
INPUT=ROOT/'outputs/analysis/rd12_exp25_temporal_coverage_20261007_r1'


def prepare(root):
    """接口：复用同一冻结200及十题输入，正式只定4秒联合重复度，旧控制另列。"""
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():return verify(root)
    old=read_json(INPUT/'protocol.json');sources=dict(old['sources']);sources[str(INPUT/'protocol.json')]=sha256(INPUT/'protocol.json')
    for path in (Path(__file__),ROOT/'src/videoqa_methods/visual_temporal_redundancy.py',ROOT/'src/videoqa_methods/weighted_seed_support.py'):
        sources[str(path)]=sha256(path)
    for q in old['question_ids']:
        p=BASE/'features'/(q+'.npy');sources[str(p)]=sha256(p)
    spec=dict(old,method=METHOD,sources=sources,temporal_scale_seconds=4.0,
        rule='Only D=max selected cos+*exp(-abs(source time difference)/4); original L, sqrt exponent, weighted visual coverage and density/DP unchanged',
        control='Time disabled, exact original exp12 prefix/selection replay before new rule',
        preparation_parent=str(INPUT),utc=utc())
    import shutil
    for path in (Path(__file__),ROOT/'src/videoqa_methods/visual_temporal_redundancy.py'):
        dest=root/'code_snapshot'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    durable(root/'protocol.json',spec);return spec


def verify(root):
    """接口：源字节绑定一套规则；恢复或后续问答不得修改尺度、公式或父池。"""
    spec=read_json(root/'protocol.json')
    for p,h in spec['sources'].items():
        if sha256(p)!=h:raise ValueError('Joint redundancy source changed: '+p)
    return spec


def analyze(root):
    """接口：200＋固定面板，先重放原纯视觉控制，再完整重算新前缀和原预算目标。"""
    spec=verify(root)
    if (root/'summary.json').exists():
        saved=read_json(root/'summary.json')
        for p,h in saved['artifacts'].items():
            if sha256(p)!=h:raise ValueError('Completed joint selection changed')
        return saved
    with exclusive(root/'select.lock'):
        records={}
        # 步骤1：原始CLS不投影、不混合，不添加任何新视频或文本模型调用。
        for n,q in enumerate(spec['question_ids'],1):
            item=spec['inputs'][q];path=root/'questions'/(q+'.json')
            if path.exists():r=read_json(path)
            else:
                f=np.load(BASE/'features'/(q+'.npy'),allow_pickle=False)
                control,_=select_visual_temporal_redundancy(item['pool'],f,item['parent'],32,None)
                expected=sorted(selected_indices(item['parent']),key=lambda i:(item['pool']['candidates'][i]['source_pts'],i))
                if control!=expected:raise ValueError('Original exp12 control replay differs')
                started=time.perf_counter();ids,trace=select_visual_temporal_redundancy(item['pool'],f,item['parent'],32,4.)
                selection=dict(video=item['pool']['video'],selected_frames=[item['pool']['candidates'][i] for i in ids]);validate_frames(selection)
                r=dict(question_id=q,method=METHOD,selected_indices=ids,selection=selection,trace=trace,
                    old_indices=selected_indices(item['parent']),original_control_indices=control,
                    protocol_sha256=sha256(root/'protocol.json'),selection_seconds=time.perf_counter()-started,new_model_calls=0,qa_prediction=None)
                durable(path,r)
            if r['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('Checkpoint protocol differs')
            records[q]=r
            if n%20==0:durable(root/'status.json',dict(status='joint_redundancy_selection',completed=n,total=len(spec['question_ids']),utc=utc()))
        # 步骤2：目录仅评价精确旧见证，新细帧不因未标注被判断为无效。
        facts=[]
        for fact in read_json(FACTS)['entries']:
            q=fact['question_id'];r=records[q];pool=spec['inputs'][q]['pool'];w=set(fact['witness_pts'])
            old=bool(w&{pool['candidates'][i]['source_pts'] for i in r['old_indices']});new=bool(w&{pool['candidates'][i]['source_pts'] for i in r['selected_indices']})
            facts.append(dict(question_id=q,fact_id=fact['fact_id'],description=fact['description'],old=old,new=new))
        if len(facts)!=75 or len({f['question_id'] for f in facts})!=42:raise ValueError('Fact denominator changed')
        summary=dict(status='completed',method=METHOD,development_count=200,question_count=len(spec['question_ids']),
            changed=sum(set(records[q]['selected_indices'])!=set(records[q]['old_indices']) for q in spec['development_ids']),
            facts=facts,old_hits=sum(f['old'] for f in facts),new_hits=sum(f['new'] for f in facts),
            gained=[(f['question_id'],f['fact_id']) for f in facts if not f['old'] and f['new']],
            lost=[(f['question_id'],f['fact_id']) for f in facts if f['old'] and not f['new']],new_model_calls=0,
            artifacts={str(p):sha256(p) for p in (root/'questions').glob('*.json')},utc=utc())
        durable(root/'summary.json',summary);print({k:v for k,v in summary.items() if k not in ('facts','artifacts')},flush=True);return summary


def export_job(job):
    """接口：同源精确解码并冻结新方法RGB；没有答案输入和二次抽帧。"""
    root,q,sources=job;root=Path(root);r=read_json(root/'questions'/(q+'.json'));path=root/'exports'/(q+'.json')
    if path.exists():
        old=read_json(path)
        if old['selected_indices']!=r['selected_indices'] or old['method']!=METHOD:raise ValueError('Export identity differs')
        return q,sha256(path)
    images,hashes=FrameProvider(r['selection']).decode()
    for image in images:image.close()
    # 步骤1：正式答案由原模型产生，资产不携带人工语义需求或历史预测。
    durable(path,dict(question_id=q,method=METHOD,selection=r['selection'],selected_indices=r['selected_indices'],
        rgb_sha256=hashes,sources=sources,configuration='M32/K32; visual-temporal redundancy tau4 with original visual coverage',diagnostic=False,
        trace_path=str(root/'questions'/(q+'.json'))))
    return q,sha256(path)


def exports(root):
    """接口：AI差异初审通过后，spawn八进程冻结200份正式输入；不重复历史问答。"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    spec=verify(root)
    if not read_json(root/'review.json').get('approved_for_development'):raise ValueError('Offline gate not passed')
    if (root/'exports/selection_manifest.json').exists():return read_json(root/'exports/selection_manifest.json')
    sources={str(root/'protocol.json'):sha256(root/'protocol.json'),str(root/'summary.json'):sha256(root/'summary.json'),str(root/'review.json'):sha256(root/'review.json')}
    jobs=[(str(root),q,dict(sources,**{str(root/'questions'/(q+'.json')):sha256(root/'questions'/(q+'.json'))})) for q in spec['development_ids']]
    # 步骤1：仅解码，不加载模型；线程与已验收精确源帧提供器保持一致。
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as executor:
        outputs=dict(executor.map(export_job,jobs))
    # 步骤2：继承同协议Qwen32基线后端，保存新方法和独立调用预算。
    manifest=dict(status='completed',method=METHOD,question_count=200,frame_budget=32,question_ids=spec['development_ids'],
        dataset_manifest=str(DEV),manifest_sha256=sha256(DEV),parent_run=str(BASE),parent_protocol_sha256=sha256(BASE/'protocol.json'),
        exports=outputs,configuration='M32/K32; joint visual-temporal redundancy tau4, original exp12 visual coverage',diagnostic=False)
    durable(root/'exports/selection_manifest.json',manifest);return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--stage',choices=('prepare','analyze','render','export'),required=True)
    args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run ID')
    root=ROOT/'outputs/analysis'/args.run_id
    try:{'prepare':prepare,'analyze':analyze,'render':render,'export':exports}[args.stage](root)
    except BaseException as exc:
        durable(root/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),utc=utc()));raise
