"""固定候选池的时间覆盖选帧准备；选帧、原帧复核和导出均不调用模型。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import argparse
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,rows,utc
from videoqa_methods.temporal_coverage import select_temporal_coverage
from videoqa_audit.coverage_objective_review import selected_indices
from videoqa_budget32.frames import FrameProvider,validate_frames
from analyze_coverage_objectives import Reader,load_pair,DEV,FACTS,BASE,Q12

METHOD='RD-1.2-exp25'
PILOT=('035-2','018-2','363-2','409-3','556-1','383-2','003-2','006-1','103-1','395-2')


def prepare(root):
    """接口：先冻结开发200及十题面板、原池/轨迹及一套公式，不读新答案选参数。"""
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():return verify(root)
    dev=[r['question_id'] for r in read_json(DEV)['rows']]
    ids=list(dict.fromkeys(dev+list(PILOT)));reader=Reader();inputs={}
    for name in ('scripts/prepare_temporal_coverage.py','src/videoqa_methods/temporal_coverage.py',
        'src/videoqa_methods/prefix_coverage_budget.py','src/videoqa_methods/density_selection.py','scripts/analyze_coverage_objectives.py'):
        reader.bind(ROOT/name)
    reader.bind(DEV);reader.bind(FACTS)
    # 步骤1：父模型/算法来自已验收全量，不混用旧8/16帧协议。
    for q in ids:
        pool,_,parent=load_pair('qwen32',q,reader)
        inputs[q]=dict(pool=pool,parent=parent)
    spec=dict(method=METHOD,parent='RD-1.2-exp12',inputs=inputs,question_ids=ids,development_ids=dev,
        sources=reader.sources,frame_budget=32,pilot=list(PILOT),new_model_calls=0,
        rule='ITM weighted normalized squared distance on observed coarse timestamps; protected-anchor greedy prefix and original density/positive-quota DP',
        freezes='Original seeds, cores, windows, refinement mappings and complete coarse/fine pool; no feature, scorer or frame budget change',
        development_qa_limit=200,full_qa_limit=2700,
        offline_gate='Input protocol/replay valid; at least two videos show additional useful state/process opportunity, not only exact catalog hits; known protection losses recorded',
        full_gate='At least140/200 versus exp12 138, supported cross-video mechanism and reviewed paired losses; exploratory resource gate, not statistical confirmation',utc=utc())
    import shutil
    for path in (ROOT/'scripts/prepare_temporal_coverage.py',ROOT/'src/videoqa_methods/temporal_coverage.py'):
        dest=root/'code_snapshot'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    durable(root/'protocol.json',spec);return spec


def verify(root):
    """接口：所有固定输入与选择代码字节一致时才恢复，不自动接受改参数。"""
    spec=read_json(root/'protocol.json')
    for p,h in spec['sources'].items():
        if sha256(p)!=h:raise ValueError('Temporal input or code changed: '+p)
    return spec


def analyze(root):
    """接口：单一规则全部200及面板重放，实际旧/新PTS选择与固定父配额诊断落盘。"""
    spec=verify(root)
    if (root/'summary.json').exists():
        saved=read_json(root/'summary.json')
        for p,h in saved['artifacts'].items():
            if sha256(p)!=h:raise ValueError('Completed selection changed')
        return saved
    with exclusive(root/'select.lock'):
        records={}
        # 步骤1：全局粗资格与细帧来源冻结，父配额只作控制，不决定正式DP。
        for n,q in enumerate(spec['question_ids'],1):
            path=root/'questions'/(q+'.json');item=spec['inputs'][q]
            if path.exists():
                r=read_json(path)
                if r['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('Selection checkpoint changed')
            else:
                started=time.perf_counter();ids,trace=select_temporal_coverage(item['pool'],item['parent'],32)
                selection=dict(video=item['pool']['video'],selected_frames=[item['pool']['candidates'][i] for i in ids])
                validate_frames(selection)
                r=dict(question_id=q,method=METHOD,selected_indices=ids,selection=selection,trace=trace,
                    old_indices=selected_indices(item['parent']),protocol_sha256=sha256(root/'protocol.json'),
                    selection_seconds=time.perf_counter()-started,new_model_calls=0,qa_prediction=None)
                durable(path,r)
            records[q]=r
            if n%20==0:durable(root/'status.json',dict(status='offline_selection',completed=n,total=len(spec['question_ids']),utc=utc()))
        facts=[]
        # 步骤2：目录是非穷尽代理，细帧没标注不自动判无关，所有得失如实列出。
        for f in read_json(FACTS)['entries']:
            q=f['question_id'];pool=spec['inputs'][q]['pool'];r=records[q];w=set(f['witness_pts'])
            old=bool(w&{pool['candidates'][i]['source_pts'] for i in r['old_indices']})
            new=bool(w&{pool['candidates'][i]['source_pts'] for i in r['selected_indices']})
            facts.append(dict(question_id=q,fact_id=f['fact_id'],description=f['description'],old=old,new=new))
        if len(facts)!=75 or len({f['question_id'] for f in facts})!=42:raise ValueError('Frozen fact denominator changed')
        selected_dev=[records[q] for q in spec['development_ids']]
        summary=dict(status='completed',method=METHOD,question_count=len(spec['question_ids']),development_count=200,
            changed=sum(set(r['selected_indices'])!=set(r['old_indices']) for r in selected_dev),
            facts=facts,old_hits=sum(f['old'] for f in facts),new_hits=sum(f['new'] for f in facts),
            gained=[(f['question_id'],f['fact_id']) for f in facts if not f['old'] and f['new']],
            lost=[(f['question_id'],f['fact_id']) for f in facts if f['old'] and not f['new']],new_model_calls=0,
            artifacts={str(p):sha256(p) for p in (root/'questions').glob('*.json')},utc=utc())
        durable(root/'summary.json',summary);print({k:v for k,v in summary.items() if k not in ('facts','artifacts')},flush=True)
        return summary


def render(root):
    """接口：固定十题的新旧原图板，精确解码/RGB身份，原问答不重跑。"""
    from analyze_coverage_objectives import image_board
    spec=verify(root)
    for q in PILOT:
        item=spec['inputs'][q];r=read_json(root/'questions'/(q+'.json'))
        for label,ids in [('parent',r['old_indices']),('temporal',r['selected_indices'])]:
            frames=sorted([item['pool']['candidates'][i] for i in ids],key=lambda f:(f['source_pts'],f['candidate_index']))
            images,hashes=FrameProvider(dict(video=item['pool']['video'],selected_frames=frames)).decode()
            dest=root/'review'/q/(label+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True)
            image_board(images,frames,dest)
            for im in images:im.close()
            durable(dest.with_suffix('.json'),dict(frames=frames,rgb_sha256=hashes,view='Source RGB thumbnails, not model processor tensors',new_model_calls=0))
    return len(PILOT)*2


def export_one(job):
    """接口：新选帧从原视频精确读取，按正式200题提供器所需格式冻结，不选择答案。"""
    root,q,sources=job;root=Path(root);item=read_json(root/'questions'/(q+'.json'))
    path=root/'exports'/(q+'.json')
    if path.exists():
        old=read_json(path)
        if old['selected_indices']!=item['selected_indices']:raise ValueError('Export selection changed')
        return q,sha256(path)
    # 步骤1：直接复用源帧提供器，不重采样、复制或调预算。
    images,hashes=FrameProvider(item['selection']).decode()
    for image in images:image.close()
    # 步骤2：问答只用原题/选项；选择追踪和源身份独立保留。
    record=dict(question_id=q,method=METHOD,selection=item['selection'],selected_indices=item['selected_indices'],
        rgb_sha256=hashes,sources=sources,configuration='M32/K32; temporal coverage over frozen exp12 candidate pool',
        diagnostic=False,trace_path=str(root/'questions'/(q+'.json')))
    durable(path,record);return q,sha256(path)


def exports(root):
    """接口：仅在AI语义初审通过后并行冻结200个输入；不依赖会话或人工再确认。"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    spec=verify(root);review=read_json(root/'review.json')
    if not review.get('approved_for_development'):raise ValueError('Offline review gate not passed')
    if (root/'exports/selection_manifest.json').exists():return read_json(root/'exports/selection_manifest.json')
    sources={str(root/'protocol.json'):sha256(root/'protocol.json'),str(root/'summary.json'):sha256(root/'summary.json'),
        str(root/'review.json'):sha256(root/'review.json')}
    # 步骤1：spawn独立解码进程，避免曾发生的线程争用与解释器首次懒导入竞态。
    jobs=[(str(root),q,dict(sources,**{str(root/'questions'/(q+'.json')):sha256(root/'questions'/(q+'.json'))})) for q in spec['development_ids']]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as executor:
        exported=dict(executor.map(export_one,jobs))
    # 步骤2：模型协议继承原Qwen32正式父运行，新增方法不覆盖历史记录。
    manifest=dict(status='completed',method=METHOD,question_count=200,frame_budget=32,question_ids=spec['development_ids'],
        dataset_manifest=str(DEV),manifest_sha256=sha256(DEV),parent_run=str(BASE),parent_protocol_sha256=sha256(BASE/'protocol.json'),
        exports=exported,configuration='M32/K32; same exp12 pool with ITM-weighted temporal quantization',diagnostic=False)
    durable(root/'exports/selection_manifest.json',manifest);return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--stage',choices=('prepare','analyze','render','export'),required=True)
    args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run ID')
    root=ROOT/'outputs/analysis'/args.run_id
    try:
        {'prepare':prepare,'analyze':analyze,'render':render,'export':exports}[args.stage](root)
    except BaseException as exc:
        durable(root/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),utc=utc()));raise
