"""三题时间字段人工诊断：最多6次真实问答，不构成选帧算法成绩。"""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,rows,utc,invoke_once,QWEN_PATH
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider,validate_frames
from videoqa_budget32.run import process,validate,recover_call,critical_code,PYTHON
from videoqa_audit.framework_contraction import BASE,RUN32
from videoqa_audit.region_scope_runtime import parent_result,cpu_identity,parent_call_sources
from videoqa_lmms_bridge.common import framework_identity

CHANGES = {
    '555-2':dict(add=[117,119],remove=[524,562],reason='保留Reality卧室语境与10:00AM时间卡，移除Amazon屏幕和片尾SUB文字'),
    '560-2':dict(add=[190],remove=[350],reason='保留外出段7:30pm字幕，移除片尾订阅提示；原打扫段4:30pm仍保留'),
    '599-1':dict(add=[21],remove=[673],reason='保留关闹钟附近正面手机时间，移除频道片尾图'),
}


def repaired_selection(old,pool,change):
    """接口：仅以冻结粗候选做至多两张人工替换，其他源帧与总预算保持不变。"""
    if not 1<=len(change['add'])==len(change['remove'])<=2: raise ValueError('Repair budget differs')
    if any(i<0 or i>=pool['initial_count'] for i in change['add']+change['remove']):
        raise ValueError('Repair must use original coarse candidates')
    frames = old['selected_frames']; add=[pool['candidates'][i] for i in change['add']]
    removed={pool['candidates'][i]['source_pts'] for i in change['remove']}
    current={f['source_pts'] for f in frames}
    if len(frames)!=32 or not removed<=current or {f['source_pts'] for f in add}&current:
        raise ValueError('Repair source identity invalid')
    selected=sorted([f for f in frames if f['source_pts'] not in removed]+add,key=lambda f:f['source_pts'])
    selection=dict(old,selected_frames=selected); validate_frames(selection,32)
    if len(selected)!=32: raise ValueError('Repair changed source frame count')
    return selection


def prepare(root):
    """接口：CPU冻结六输入、正式处理像素和提示Token，原历史输入必须逐项复现。"""
    root=Path(root); root.mkdir(parents=True,exist_ok=True); offline_environment()
    with exclusive(root/'prepare.lock'):
        if (root/'protocol.json').exists(): return verify(root)
        import torch
        from PIL import Image
        from transformers import AutoProcessor
        from videoqa_budget32.observe import tensor_hash
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        source_spec=read_json(RUN32/'protocol.json')
        if framework_identity()!=source_spec['framework'] or environment(PYTHON)!=source_spec['environment']:
            raise ValueError('Parent evaluation environment changed')
        for path,meta in source_spec['model_files'].items():
            if sha256(path)!=meta['sha256']: raise ValueError('Frozen model bytes changed')
        processor=AutoProcessor.from_pretrained(QWEN_PATH,local_files_only=True,use_fast=False,
            min_pixels=100352,max_pixels=602112,size=dict(shortest_edge=100352,longest_edge=602112))
        data={r['question_id']:r for r in rows()}; sources={str(RUN32/'protocol.json'):sha256(RUN32/'protocol.json')}
        files={}; observations={}
        # 步骤1：人工见证只决定诊断输入，不进入原问题/选项/提示词，不做答案检索。
        for q,change in CHANGES.items():
            path=parent_result('qwen32',q); old=read_json(path); pp=BASE/'pools'/(q+'.json'); pool=read_json(pp)
            sources.update(parent_call_sources('qwen32',path,old)); sources[str(pp)]=sha256(pp)
            video=Path(old['selection']['video']['path']); expected=source_spec['video_hashes'][data[q]['video_id']]
            if sha256(video)!=expected: raise ValueError('Video bytes changed')
            repaired=repaired_selection(old['selection'],pool,change)
            observations[q]=dict(historical_result=str(path),historical_prediction=old['answer']['parsed_answer'],
                                 changes=change,conditions={})
            # 步骤2：模型输入身份在调用前核验，实际新增帧从官方像素Tensor逆排供复核。
            for condition,selection in [('original',old['selection']),('repair',repaired)]:
                images,rgb=FrameProvider(selection).decode()
                try:
                    identity=cpu_identity('qwen32',(processor,None),images,data[q])
                    if condition=='original':
                        obs=old['answer']['observed']
                        if (rgb!=old['rgb_sha256'] or identity['pixel_sha256']!=obs['processor_pixel_sha256']
                                or identity['prompt']!=obs['prompt'] or identity['input_token_ids']!=obs['input_token_ids']
                                or identity['video_grid_thw']!=obs['video_grid_thw']):
                            raise ValueError('Historical actual input not reproduced')
                    rel=f'selections/{condition}/{q}.json'
                    durable(root/rel,dict(question_id=q,method=condition,selection=selection,rgb_sha256=rgb,input_identity=identity))
                    files[rel]=sha256(root/rel); observations[q]['conditions'][condition]=identity
                    if condition=='repair':
                        ip=processor.image_processor
                        x=ip(images=None,videos=[np.stack([np.asarray(im.convert('RGB')) for im in images])],
                            size=dict(shortest_edge=100352,longest_edge=602112),return_tensors='pt')
                        if tensor_hash(x['pixel_values_videos'])!=identity['pixel_sha256']:
                            raise ValueError('Visibility view not actual processor pixels')
                        t,h,w=x['video_grid_thw'].tolist()[0]
                        pixels=x['pixel_values_videos'].numpy().reshape(t,h//2,w//2,2,2,3,2,14,14).transpose(0,6,5,1,3,7,2,4,8).reshape(t*2,3,h*14,w*14)
                        additions={pool['candidates'][i]['source_pts'] for i in change['add']}
                        for i,frame in enumerate(selection['selected_frames']):
                            if frame['source_pts'] not in additions: continue
                            visible=np.clip((pixels[i].transpose(1,2,0)*np.asarray(ip.image_std)+np.asarray(ip.image_mean))*255,0,255).round().astype(np.uint8)
                            view=root/'visibility'/(q+'_'+str(frame['source_pts'])+'.png')
                            view.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(visible).save(view)
                            files[str(view.relative_to(root))]=sha256(view)
                finally:
                    for im in images: im.close()
        # 步骤3：独立六次额度事前冻结，与此前022两次诊断账本不混算。
        code=dict(critical_code()); code[str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
        spec=dict(model=source_spec['model'],question_ids=list(CHANGES),methods=['original','repair'],
            max_calls=6,max_calls_per_method=3,allowed_gpus=list(map(str,range(8))),code_sha256=code,
            model_files=source_spec['model_files'],framework=source_spec['framework'],environment=source_spec['environment'],
            video_stats={data[q]['video_id']:source_spec['video_stats'][data[q]['video_id']] for q in CHANGES},
            selection_exports={p:h for p,h in files.items() if p.startswith('selections/')},all_input_exports=files,
            sources=sources,observations=observations,max_synthetic_calls=1,
            scope='Human-directed time-field and event-context replacement; not algorithm accuracy',utc=utc())
        durable(root/'protocol.json',spec); return spec


def verify(root):
    """接口：检查输入/代码/环境，未知生成状态不允许通过重跑修复。"""
    root=Path(root); spec=read_json(root/'protocol.json')
    if spec['question_ids']!=list(CHANGES) or spec['max_calls']!=6 or spec['max_calls_per_method']!=3:
        raise ValueError('Diagnostic call population changed')
    if framework_identity()!=spec['framework'] or environment(PYTHON)!=spec['environment']:
        raise ValueError('Diagnostic environment changed')
    for p,h in spec['sources'].items():
        if sha256(p)!=h: raise ValueError('Diagnostic input source changed')
    for p,h in spec['all_input_exports'].items():
        if sha256(root/p)!=h: raise ValueError('Frozen diagnostic input changed')
    for p,h in spec['code_sha256'].items():
        if sha256(ROOT/p)!=h: raise ValueError('Diagnostic code changed')
    for p,v in spec['model_files'].items():
        st=Path(p).stat()
        if st.st_size!=v['bytes'] or st.st_mtime_ns!=v['mtime_ns']: raise ValueError('Model identity changed')
    return spec


def audit(root):
    """接口：独立复算六答案，历史复测不一致只暂停对应题的归因。"""
    root=Path(root); spec=verify(root); bank={r['question_id']:r for r in rows()}; pairs=[]
    for q in CHANGES:
        a=validate(root,bank[q],'original',spec); b=validate(root,bank[q],'repair',spec)
        for condition,r in [('original',a),('repair',b)]:
            o=r['answer']['observed']; expected=spec['observations'][q]['conditions'][condition]
            if (o['processor_pixel_sha256']!=expected['pixel_sha256'] or o['prompt']!=expected['prompt']
                    or o['input_token_ids']!=expected['input_token_ids'] or o['video_grid_thw']!=expected['video_grid_thw']):
                raise ValueError('Observed official input differs')
        calls=[read_json(root/'calls'/c/(q+'.json')) for c in ('original','repair')]
        if calls[0]['pid']!=calls[1]['pid'] or calls[0]['gpu']!=calls[1]['gpu']:
            raise ValueError('Pair was not same process/GPU')
        replay=a['answer']['parsed_answer']==spec['observations'][q]['historical_prediction']
        pairs.append(dict(question_id=q,original_prediction=a['answer']['parsed_answer'],repair_prediction=b['answer']['parsed_answer'],
            original_correct=a['correct'],repair_correct=b['correct'],historical_replay_matches=replay,causal_attribution_allowed=replay))
    paths=list((root/'results').glob('*/*.json'))+list((root/'calls').glob('*/*.json'))+list((root/'raw_generations').glob('*/*.json'))
    if len(paths)!=18: raise ValueError('Six physical results/calls/raw returns not closed')
    value=dict(status='completed',real_calls=6,synthetic_calls=1,pairs=pairs,
        artifacts={str(p):sha256(p) for p in paths},scope=spec['scope'],utc=utc())
    old=root/'summary.json'
    if old.exists() and read_json(old)['artifacts']!=value['artifacts']: raise ValueError('Completed files changed on recovery')
    durable(old,value); durable(root/'status.json',value); return value


def execute(root):
    """接口：后台用一张空闲卡执行六诊断，合成单列；不按新答案追加输入。"""
    root=Path(root); spec=verify(root)
    if not (root/'visibility_review.json').exists() or read_json(root/'visibility_review.json')['protocol_sha256']!=sha256(root/'protocol.json'):
        raise ValueError('Actual-input visibility not reviewed')
    if read_json(root/'visibility_review.json')['status']!='passed': raise ValueError('Visibility gate not passed')
    with exclusive(root/'coordinator.lock'):
        bank={r['question_id']:r for r in rows()}; ordinal={r['question_id']:i for i,r in enumerate(rows())}
        schedule=[(q,c) for q in CHANGES for c in (['original','repair'] if ordinal[q]%2==0 else ['repair','original'])]
        calls=list((root/'calls').glob('*/*.json'))
        if len(calls)>6 or any((p.stem,p.parent.name) not in schedule for p in calls): raise ValueError('Call budget invalid')
        for p in calls: recover_call(root,bank[p.stem],p.parent.name,spec)
        if all((root/'results'/c/(q+'.json')).exists() for q,c in schedule): return audit(root)
        if calls: raise RuntimeError('Incomplete detached pair process; inspect raw returns, no new generation')
        from videoqa_runtime.baseline_run import available_gpus
        while True:
            try: free,_=available_gpus(spec['allowed_gpus'])
            except RuntimeError: free=[]
            if free: break
            durable(root/'status.json',dict(status='waiting_resources',pid=os.getpid(),sid=os.getsid(0),utc=utc())); time.sleep(15)
        gpu=free[0]; os.environ['CUDA_VISIBLE_DEVICES']=gpu; offline_environment()
        import torch
        from PIL import Image
        from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
        from videoqa_budget32.qwen_adapter import FrozenInputQwen
        from videoqa_budget32.observe import observe
        from videoqa_lmms_bridge.bridge import synthetic_row
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        backend=QwenFA2VideoBackend(); adapter=FrozenInputQwen(backend)
        durable(root/'load_report.json',dict(gpu=gpu,pid=os.getpid(),report=backend.load_report))
        images=[Image.new('RGB',(896,672),(40,i*7,150)) for i in range(32)]
        fp=sha256(root/'protocol.json')
        try:
            invoke_once(root/'synthetic','warmup','synthetic',fp,gpu,lambda state: observe(backend,adapter,images,synthetic_row(32),state,
                root/'synthetic_raw.json',dict(question_id='warmup',method='synthetic',protocol_sha256=fp,gpu=gpu)))
        finally:
            for im in images: im.close()
        # 步骤1：调用前再次核验实际处理器；只有原问题及原选项进入模型。
        for q,c in schedule:
            durable(root/'status.json',dict(status='running',question_id=q,condition=c,gpu=gpu,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            item=read_json(root/'selections'/c/(q+'.json')); images,rgb=FrameProvider(item['selection']).decode()
            try:
                if rgb!=item['rgb_sha256'] or cpu_identity('qwen32',(backend.processor,None),images,bank[q])!=item['input_identity']:
                    raise ValueError('Actual frozen input changed before generation')
            finally:
                for im in images: im.close()
            process(root,bank[q],c,backend,adapter,gpu,spec)
        # 步骤2：计分与诊断独立审计；改善只能解释人工组合干预，不升级任何方法。
        return audit(root)


def main():
    """接口：准备、后台问答与完成恢复分阶段，不启动开发200题或全量算法测试。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=('prepare','qa','audit'),required=True);a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    root=ROOT/'outputs/diagnostics'/a.run_id
    try:
        (prepare if a.stage=='prepare' else execute if a.stage=='qa' else audit)(root)
    except BaseException as exc:
        durable(root/'failure.json',dict(status='paused',error=repr(exc),utc=utc()));raise


if __name__=='__main__': main()
