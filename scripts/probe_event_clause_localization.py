"""原文事件子句定位对照：五题实际竞争语境，零问答，不改字段规则或正式选帧。"""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import numpy as np

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,rows,utc
from videoqa_full.prepare import environment
from videoqa_audit.framework_contraction import BASE
from videoqa_audit.event_clause import event_clause
from videoqa_audit.query_input_probe import score_once,MODEL,PYTHON
from videoqa_budget32.selection import SOURCE,configure_visual_precision
from videoqa_budget32.frames import FrameProvider
from videoqa_methods.density_selection import source_time

FIELD=ROOT/'outputs/analysis/explicit_time_fields__videomme19__20261009_r1'
TARGETS={'555-2':119,'560-2':190,'599-1':21,'252-3':55,'553-3':511}
FILES=('scripts/probe_event_clause_localization.py','src/videoqa_audit/event_clause.py',
 'src/videoqa_audit/query_input_probe.py','src/videoqa_runtime/baseline_selection.py',
 'src/videoqa_methods/model_ops.py','src/videoqa_budget32/selection.py','src/videoqa_budget32/frames.py')


def prepare(root):
    """接口：先冻结真实目标/前4竞争语境银行，再分别评分；人工标签不进入评分器。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'prepare.lock'):
        if (root/'protocol.json').exists():return verify(root)
        if read_json(FIELD/'status.json')['pilot_inventory_gate'] is not False:
            raise ValueError('Prior fixed field branch not closed as expected')
        data={r['question_id']:r for r in rows()};sources={};inputs={};positions=0;batches=0
        old=read_json(SOURCE/'protocol.json')
        model_files={p:v['sha256'] for p,v in old['model_files'].items() if p.startswith(str(MODEL)+'/')}
        if not model_files:raise ValueError('Frozen BLIP source identity missing')
        for p,h in model_files.items():
            if sha256(p)!=h:raise ValueError('BLIP bytes changed')
        # 步骤1：银行只是实际竞争诊断，不用于生成正式候选或声明全量检索收益。
        for q,target in TARGETS.items():
            pp=BASE/'pools'/(q+'.json');fp=FIELD/'proposals'/(q+'.json');pool=read_json(pp);proposals=read_json(fp)
            c=pool['candidates'][:pool['initial_count']];video=pool['video'];t=source_time(c[target],video)
            groups=[dict(name='listed_target_context',target=True,indices=[i for i,f in enumerate(c) if abs(source_time(f,video)-t)<=4])]
            for p in proposals['proposals'][:4]:
                ids=sorted({i for r in p['regions'] for i in r['context_indices']})
                if any(g['indices']==ids for g in groups):continue
                groups.append(dict(name='prior_proposal_'+str(p['rank']),target=False,indices=ids))
            indices=sorted({i for g in groups for i in g['indices']});query=event_clause(data[q]['question'])
            item=dict(question_id=q,pool_path=str(pp),indices=indices,groups=groups,query=query,
                      conditions=dict(original_question=query['original_question'],event_clause=query['event_query']))
            path=root/'inputs'/(q+'.json');durable(path,item);inputs[q]=sha256(path)
            sources.update({str(pp):sha256(pp),str(fp):sha256(fp)})
            positions+=len(indices);batches+=2*((len(indices)+15)//16)
        # 步骤2：固定两查询，不添加同义词、选项或未知事实；原题重放不符停止。
        names={n:sha256(ROOT/n) for n in FILES}
        spec=dict(reference_method='RD-1.2-exp07',question_ids=list(TARGETS),input_exports=inputs,sources=sources,
            model_files=model_files,files=names,environment=environment(PYTHON),allowed_gpus=list(map(str,range(8))),
            selected_context_frame_positions=positions,frame_text_scores=2*positions,real_batch_calls=batches,
            synthetic_batch_calls=1,new_qa_calls=0,new_query_generation_calls=0,batch_size=16,replay_tolerance=1e-6,
            scope='Fixed actual-competition bank; event-only actor ablation is not a deployed query replacement',utc=utc())
        import shutil
        for n in FILES:
            p=root/'code_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
        durable(root/'protocol.json',spec);return spec


def verify(root):
    """接口：来源、依赖、查询或模型身份变化暂停；完成批次不重新前向。"""
    root=Path(root);spec=read_json(root/'protocol.json')
    if spec['question_ids']!=list(TARGETS) or spec['new_qa_calls']!=0:raise ValueError('Probe scope changed')
    if environment(PYTHON)!=spec['environment']:raise ValueError('BLIP environment changed')
    for p,h in {**spec['sources'],**spec['model_files']}.items():
        if sha256(p)!=h:raise ValueError('Probe source changed')
    for n,h in spec['files'].items():
        if sha256(ROOT/n)!=h:raise ValueError('Probe code changed')
    for q,h in spec['input_exports'].items():
        if sha256(root/'inputs'/(q+'.json'))!=h:raise ValueError('Locating queries changed')
    return spec


def report(root):
    """接口：只在同一查询内部比较各语境的相对排名，不跨查询比较绝对概率。"""
    root=Path(root);spec=verify(root);results=[]
    for q in TARGETS:
        item=read_json(root/'inputs'/(q+'.json'));positions={i:j for j,i in enumerate(item['indices'])};values={}
        for condition in item['conditions']:
            scores=[]
            for start in range(0,len(item['indices']),16):
                call=read_json(root/'calls'/q/condition/(str(start)+'.json'))
                if call['status']!='completed' or call['identity']['protocol_sha256']!=sha256(root/'protocol.json'):
                    raise ValueError('BLIP batch not confirmed')
                scores+=call['scores']
            groups=[dict(g,score=max(scores[positions[i]] for i in g['indices'])) for g in item['groups']]
            target=groups[0]['score'];rank=1+sum(g['score']>target for g in groups[1:])
            values[condition]=dict(groups=groups,target_rank=rank)
        results.append(dict(question_id=q,queries=item['query'],comparisons=values,
            rank_improved=values['event_clause']['target_rank']<values['original_question']['target_rank']))
    main=sum(x['rank_improved'] for x in results[:3]);controls=all(x['comparisons']['event_clause']['target_rank']<=x['comparisons']['original_question']['target_rank'] for x in results[3:])
    value=dict(status='completed',results=results,main_rank_improvements=main,controls_non_degrading=controls,
        locating_signal_supported=main>=2 and controls,new_qa_calls=0,
        limitation='Human-constructed fixed bank, actor information temporarily ablated; not global recall or QA accuracy')
    durable(root/'summary.json',value);durable(root/'status.json',dict(status='completed',utc=utc()));return value


def execute(root):
    """接口：一卡长期BLIP，原题/事件查询共用同批RGB；既有原题分数误差须<1e-6。"""
    root=Path(root);spec=verify(root)
    if (root/'summary.json').exists():return report(root)
    with exclusive(root/'coordinator.lock'):
        from videoqa_runtime.baseline_run import available_gpus
        while True:
            try:gpus,_=available_gpus(spec['allowed_gpus'])
            except RuntimeError:gpus=[]
            if gpus:break
            durable(root/'status.json',dict(status='waiting_resources',pid=os.getpid(),sid=os.getsid(0),utc=utc()));time.sleep(15)
        gpu=gpus[0];os.environ['CUDA_VISIBLE_DEVICES']=gpu;offline_environment()
        import torch
        from PIL import Image
        from videoqa_methods.model_ops import FeatureScorer
        torch.set_num_threads(2);torch.set_num_interop_threads(1);precision=configure_visual_precision(torch)
        scorer=FeatureScorer(str(MODEL));images=[Image.new('RGB',(1280,720),'gray') for _ in range(16)]
        try:score_once(root,'synthetic','warmup',0,scorer,scorer.tokenize('Which synthetic color is visible?'),images,gpu,
                       [hashlib.sha256(np.asarray(i).tobytes()).hexdigest() for i in images])
        finally:
            for image in images:image.close()
        # 步骤1：只对冻结银行前向，不改源图，不把目标标签传给模型。
        for q in TARGETS:
            item=read_json(root/'inputs'/(q+'.json'));pool=read_json(item['pool_path']);indices=item['indices']
            encoded={name:scorer.tokenize(text) for name,text in item['conditions'].items()}
            for start in range(0,len(indices),16):
                batch=indices[start:start+16];selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in batch])
                images,rgb=FrameProvider(selection).decode()
                try:
                    original,_=score_once(root,q,'original_question',start,scorer,encoded['original_question'],images,gpu,rgb)
                    error=max(abs(a-pool['initial_scores'][i]) for a,i in zip(original,batch))
                    if error>spec['replay_tolerance']:raise ValueError('Original score replay differs')
                    score_once(root,q,'event_clause',start,scorer,encoded['event_clause'],images,gpu,rgb)
                finally:
                    for image in images:image.close()
                durable(root/'status.json',dict(status='scoring',question_id=q,completed_frames=start+len(batch),total=len(indices),
                    gpu=gpu,pid=os.getpid(),sid=os.getsid(0),precision=precision,utc=utc()))
        # 步骤2：确认相对排序后停止本批，无论结果如何都不直接执行新200题。
        return report(root)


def main():
    """接口：独立事件定位能力检验；字段先行失败结果保持原样，不扫描查询模板。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=('prepare','score','report'),required=True);a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    root=ROOT/'outputs/analysis'/a.run_id
    try:(prepare if a.stage=='prepare' else execute if a.stage=='score' else report)(root)
    except BaseException as exc:
        durable(root/'status.json',dict(status='paused',error=repr(exc),utc=utc()));raise


if __name__=='__main__':main()
