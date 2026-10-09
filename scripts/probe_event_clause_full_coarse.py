"""相同原文事件子句的完整粗池检验；复用已确认分数，不修改正式方法或问答。"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from probe_event_clause_localization import verify,TARGETS,MODEL,PYTHON
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import score_once
from videoqa_budget32.selection import configure_visual_precision,config
from videoqa_budget32.frames import FrameProvider
from videoqa_methods.density_selection import build_regions,source_time,frame_key

PARENT=ROOT/'outputs/analysis/event_clause_context_probe__videomme5__20261009_r2'


def prepare(root):
    """接口：固定五个完整粗池及相同查询，事前冻结复用/新增评分划分。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'prepare.lock'):
        if (root/'protocol.json').exists():return verify(root)
        old=verify(PARENT)
        if not read_json(PARENT/'summary.json')['locating_signal_supported']:raise ValueError('Prior locating gate not passed')
        exports={};schedules={};imports={};new_batches=0;new_positions=0;total_coarse=0
        # 步骤1：旧195帧银行只保留为评价信息；本轮评分覆盖完整初始粗池。
        for q in TARGETS:
            source=PARENT/'inputs'/(q+'.json');item=read_json(source);pool=read_json(item['pool_path']);n=pool['initial_count']
            dest=root/'inputs'/(q+'.json');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);exports[q]=sha256(dest)
            starts=list(range(0,n,16));refs=sorted(set([0,(n-1)//16*16]));schedules[q]=dict(event_clause=starts,original_question=refs)
            imports[q]={};total_coarse+=n
            for condition,requested in schedules[q].items():
                imports[q][condition]={}
                for start in requested:
                    p=PARENT/'calls'/q/condition/(str(start)+'.json')
                    if p.exists():
                        c=read_json(p);f=p.with_suffix('.npy')
                        if c['status']!='completed' or c['identity']['frames']!=min(16,n-start) or sha256(f)!=c['feature_sha256']:
                            raise ValueError('Parent native scoring return uncertain')
                        imports[q][condition][str(start)]=dict(path=str(p),sha256=sha256(p),feature_path=str(f),feature_sha256=sha256(f))
                    else:new_batches+=1;new_positions+=min(16,n-start)
        # 步骤2：正常新前向与历史复用分别记账，最终QA/查询生成上限仍为0。
        name=str(Path(__file__).relative_to(ROOT));files=dict(old['files'],**{name:sha256(Path(__file__))})
        spec=dict(old,input_exports=exports,files=files,schedules=schedules,imports=imports,
            total_coarse_positions=total_coarse,new_batch_calls=new_batches,new_frame_text_scores=new_positions,
            parent_protocol_sha256=sha256(PARENT/'protocol.json'),frame_text_scores=new_positions,real_batch_calls=new_batches,
            numerical_adjustment='Original coarse16 blocks; completed identical batches imported, no artificial repeated candidates',
            configuration=config(32,32),scope='Complete coarse population of same five questions; event query actor ablation still not deployed')
        spec['sources']=dict(old['sources'],**{str(PARENT/'protocol.json'):sha256(PARENT/'protocol.json')})
        for n in files:
            p=root/'code_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
        durable(root/'protocol.json',spec);return spec


def returned(root,q,condition,start,spec):
    """接口：来源完整相同才引用旧分数；引用没有新调用账本或伪造性能。"""
    imported=spec['imports'][q][condition].get(str(start))
    if imported:
        if sha256(imported['path'])!=imported['sha256'] or sha256(imported['feature_path'])!=imported['feature_sha256']:
            raise ValueError('Imported score changed')
        return read_json(imported['path'])
    path=Path(root)/'calls'/q/condition/(str(start)+'.json');c=read_json(path)
    if c['status']!='completed' or c['identity']['protocol_sha256']!=sha256(Path(root)/'protocol.json') or sha256(path.with_suffix('.npy'))!=c['feature_sha256']:
        raise ValueError('New scoring return unconfirmed')
    return c


def report(root):
    """接口：全片排名/Top32/旧规则窗口可达与局部银行分开；不报告算法精度。"""
    root=Path(root);spec=verify(root);results=[];artifacts={};fresh=0;frames=0
    for q,target in TARGETS.items():
        item=read_json(root/'inputs'/(q+'.json'));pool=read_json(item['pool_path']);c=pool['candidates'][:pool['initial_count']];events=[]
        for condition,starts in spec['schedules'][q].items():
            for start in starts:
                call=returned(root,q,condition,start,spec)
                if condition=='original_question' and max(abs(s-pool['initial_scores'][start+i]) for i,s in enumerate(call['scores']))>spec['replay_tolerance']:
                    raise ValueError('Coarse score numerical control differs')
                if condition=='event_clause':events+=call['scores']
                if str(start) not in spec['imports'][q][condition]:
                    p=root/'calls'/q/condition/(str(start)+'.json');artifacts[str(p)]=sha256(p);artifacts[str(p.with_suffix('.npy'))]=sha256(p.with_suffix('.npy'))
                    fresh+=1;frames+=len(call['scores'])
        if len(events)!=len(c):raise ValueError('Incomplete full coarse scores')
        data={};target_context=item['groups'][0]['indices']
        # 步骤1：仅在各查询内部比较名次，真实PTS及源编号作为跨方案身份。
        for condition,scores in [('original_question',pool['initial_scores']),('event_clause',events)]:
            ranked=sorted(range(len(c)),key=lambda i:(-scores[i],*frame_key(c[i])));plan=build_regions(c,scores,pool['video'],spec['configuration'])
            hit=lambda i:any(source_time(c[i],pool['video'])>=__import__('fractions').Fraction(r['left_fraction']) and source_time(c[i],pool['video'])<=__import__('fractions').Fraction(r['right_fraction']) for r in plan['regions'])
            context_best=min(target_context,key=lambda i:(-scores[i],*frame_key(c[i])))
            data[condition]=dict(target_carrier_rank=ranked.index(target)+1,best_target_context_index=context_best,
                best_target_context_rank=ranked.index(context_best)+1,carrier_is_seed=target in plan['seed_indices'],
                carrier_in_window=hit(target),context_seed_count=len(set(target_context)&set(plan['seed_indices'])),
                top32=plan['seed_indices'],plan=plan,scores=scores)
        value=dict(question_id=q,query=item['query'],coarse_count=len(c),comparisons=data,new_qa_calls=0,
            scope='Listed context/carrier reachability, not semantic event precision or answer sufficiency')
        path=root/'results'/(q+'.json')
        if path.exists() and read_json(path)!=value:raise ValueError('Full scoring result changed on recovery')
        if not path.exists():durable(path,value)
        results.append(dict(question_id=q,original={k:v for k,v in data['original_question'].items() if k not in ('scores','plan')},
            event={k:v for k,v in data['event_clause'].items() if k not in ('scores','plan')}))
    if fresh!=spec['new_batch_calls'] or frames!=spec['new_frame_text_scores']:raise ValueError('New scoring budget not closed')
    # 步骤2：保留所有得失，不因字段已在窗内便称为补查或最终选帧成功。
    summary=dict(status='completed',results=results,new_batches=fresh,new_frame_text_scores=frames,artifacts=artifacts,
        new_qa_calls=0,scope='Whole five coarse pools, not modified method accuracy; actor constraints not yet recombined')
    p=root/'summary.json'
    if p.exists() and read_json(p)['artifacts']!=artifacts:raise ValueError('Completed full scoring files changed')
    durable(p,summary);durable(root/'status.json',dict(status='completed',utc=utc()));return summary


def execute(root):
    """接口：只有尚未计算的原历史批次才前向；一次GPU进程，完成状态不再加载模型。"""
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
        torch.set_num_threads(2);torch.set_num_interop_threads(1);configure_visual_precision(torch);scorer=FeatureScorer(str(MODEL))
        images=[Image.new('RGB',(1280,720),'gray') for _ in range(16)]
        try:score_once(root,'synthetic','warmup',0,scorer,scorer.tokenize('Which synthetic color is visible?'),images,gpu,[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in images])
        finally:
            for image in images:image.close()
        # 步骤1：按旧分块保持批量形状；相同输入的父分数不重新前向。
        for q in TARGETS:
            item=read_json(root/'inputs'/(q+'.json'));pool=read_json(item['pool_path']);encoded={k:scorer.tokenize(t) for k,t in item['conditions'].items()}
            for start in spec['schedules'][q]['event_clause']:
                needed=[condition for condition,starts in spec['schedules'][q].items() if start in starts and str(start) not in spec['imports'][q][condition]]
                if not needed:continue
                batch=pool['candidates'][start:min(start+16,pool['initial_count'])];images,rgb=FrameProvider(dict(video=pool['video'],selected_frames=batch)).decode()
                try:
                    for condition in needed:
                        scores,_=score_once(root,q,condition,start,scorer,encoded[condition],images,gpu,rgb)
                        if condition=='original_question' and max(abs(s-pool['initial_scores'][start+i]) for i,s in enumerate(scores))>spec['replay_tolerance']:
                            raise ValueError('Native coarse replay differs')
                finally:
                    for image in images:image.close()
                durable(root/'status.json',dict(status='scoring',question_id=q,coarse_start=start,coarse_total=pool['initial_count'],gpu=gpu,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        return report(root)


def main():
    """接口：同一事件文本完整粗池扩展，不增加新表达或自动启动问答。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=('prepare','score','report'),required=True);a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    root=ROOT/'outputs/analysis'/a.run_id
    try:(prepare if a.stage=='prepare' else execute if a.stage=='score' else report)(root)
    except BaseException as exc:
        durable(root/'status.json',dict(status='paused',error=repr(exc),utc=utc()));raise


if __name__=='__main__':main()
