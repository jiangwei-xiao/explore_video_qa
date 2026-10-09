"""事件定位对照的原批次补充验收；修复数值比较域，不修改查询或排名准则。"""
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
from videoqa_budget32.selection import configure_visual_precision
from videoqa_budget32.frames import FrameProvider

PARENT=ROOT/'outputs/analysis/event_clause_context_probe__videomme5__20261009_r1'


def prepare(root):
    """接口：原195观察帧及查询保持不变，只补齐其原历史批次成员。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'prepare.lock'):
        if (root/'protocol.json').exists():return verify(root)
        original=verify(PARENT);schedules={};positions=0;batches=0;inputs={}
        # 步骤1：旁证从完整粗编号原分块取得，不选新目标，不改变语境成员。
        for q in TARGETS:
            src=PARENT/'inputs'/(q+'.json');item=read_json(src);pool=read_json(item['pool_path'])
            starts=sorted({i//16*16 for i in item['indices']});schedules[q]=starts
            positions+=sum(min(16,pool['initial_count']-s) for s in starts);batches+=2*len(starts)
            out=root/'inputs'/(q+'.json');out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,out);inputs[q]=sha256(out)
        spec=dict(original,input_exports=inputs,native_batch_starts=schedules,frame_text_scores=2*positions,
            real_batch_calls=batches,normal_context_positions=original['selected_context_frame_positions'],
            parent_protocol_sha256=sha256(PARENT/'protocol.json'),numerical_adjustment='Original coarse16 blocks; extra members never enter comparison regions')
        spec['sources']=dict(original['sources'],**{str(PARENT/'protocol.json'):sha256(PARENT/'protocol.json')})
        name=str(Path(__file__).relative_to(ROOT));spec['files']=dict(original['files'],**{name:sha256(Path(__file__))})
        for name in spec['files']:
            out=root/'code_snapshot'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,out)
        durable(root/'protocol.json',spec);return spec


def report(root):
    """接口：只在原语境195帧上复算相对排名，补批旁证帧不能扩充观察量。"""
    root=Path(root);spec=verify(root);results=[];artifacts={}
    for q in TARGETS:
        item=read_json(root/'inputs'/(q+'.json'));pool=read_json(item['pool_path']);values={}
        for condition in item['conditions']:
            scores={}
            for start in spec['native_batch_starts'][q]:
                cp=root/'calls'/q/condition/(str(start)+'.json');call=read_json(cp);feature=cp.with_suffix('.npy')
                if call['status']!='completed' or call['identity']['protocol_sha256']!=sha256(root/'protocol.json') or sha256(feature)!=call['feature_sha256']:
                    raise ValueError('Native batch return uncertain')
                scores.update({start+i:s for i,s in enumerate(call['scores'])});artifacts[str(cp)]=sha256(cp);artifacts[str(feature)]=sha256(feature)
            if condition=='original_question' and max(abs(scores[i]-pool['initial_scores'][i]) for i in item['indices'])>spec['replay_tolerance']:
                raise ValueError('Native replay still differs')
            groups=[dict(g,score=max(scores[i] for i in g['indices'])) for g in item['groups']]
            target=groups[0]['score'];values[condition]=dict(groups=groups,target_rank=1+sum(g['score']>target for g in groups[1:]))
        results.append(dict(question_id=q,queries=item['query'],comparisons=values,
            rank_improved=values['event_clause']['target_rank']<values['original_question']['target_rank']))
    main=sum(x['rank_improved'] for x in results[:3]);controls=all(x['comparisons']['event_clause']['target_rank']<=x['comparisons']['original_question']['target_rank'] for x in results[3:])
    value=dict(status='completed',results=results,main_rank_improvements=main,controls_non_degrading=controls,
        locating_signal_supported=main>=2 and controls,new_qa_calls=0,artifacts=artifacts,
        actual_frame_text_scores=spec['frame_text_scores'],extra_frame_text_positions=spec['frame_text_scores']-2*spec['normal_context_positions'],
        limitation='Fixed human-constructed bank and temporary actor ablation, not global recall or QA gain')
    p=root/'summary.json'
    if p.exists() and read_json(p)['artifacts']!=artifacts:raise ValueError('Completed native probe changed')
    durable(p,value);durable(root/'status.json',dict(status='completed',utc=utc()));return value


def execute(root):
    """接口：历史原分块完成原题数值验收，同批评分事件查询；不放宽容差。"""
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
            for im in images:im.close()
        # 步骤1：按原分块解码；所有ROI帧与旁证帧均保留自己的源身份。
        for q in TARGETS:
            item=read_json(root/'inputs'/(q+'.json'));pool=read_json(item['pool_path']);encoded={k:scorer.tokenize(t) for k,t in item['conditions'].items()}
            for start in spec['native_batch_starts'][q]:
                ids=list(range(start,min(start+16,pool['initial_count'])));selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in ids])
                images,rgb=FrameProvider(selection).decode()
                try:
                    original,_=score_once(root,q,'original_question',start,scorer,encoded['original_question'],images,gpu,rgb)
                    if max(abs(s-pool['initial_scores'][i]) for s,i in zip(original,ids))>spec['replay_tolerance']:
                        raise ValueError('Exact historical batch differs; stop')
                    score_once(root,q,'event_clause',start,scorer,encoded['event_clause'],images,gpu,rgb)
                finally:
                    for im in images:im.close()
                durable(root/'status.json',dict(status='scoring',question_id=q,native_start=start,gpu=gpu,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤2：只复算原ROI目标及竞争区，额外旁证不产生新排名机会。
        return report(root)


def main():
    """接口：独立补充运行，不修改原暂停记录、查询、观察银行或实验参数。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=('prepare','score','report'),required=True);a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    root=ROOT/'outputs/analysis'/a.run_id
    try:(prepare if a.stage=='prepare' else execute if a.stage=='score' else report)(root)
    except BaseException as exc:
        durable(root/'status.json',dict(status='paused',error=repr(exc),utc=utc()));raise


if __name__=='__main__':main()
