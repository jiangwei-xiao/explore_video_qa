"""当前exp12中对象展示与报社上下文的两题人工诊断输入准备。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,QWEN_PATH
from videoqa_budget32.frames import FrameProvider,validate_frames
from videoqa_budget32.observe import tensor_hash
from videoqa_audit.query_input_probe import BASE

RUN=ROOT/'outputs/diagnostics/context_repairs_exp12_20261004_r1'
PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
CHANGES={
    '541-1':dict(add=[99,100],remove=[178,179],
                 rationale='加入棕色袋展示及袋口化妆品，移除两张相邻眼罩展开画面；不声称移除帧无用'),
    '705-1':dict(add=[2373,2379],remove=[781,808],
                 rationale='加入报社场景与首张署名文章，移除两张相近会话人物画面；不单独归因重复或去干扰'),
}


def prepare():
    """接口：只准备四份同预算输入及实际处理图，问答前必须完成可见性审核。"""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    with exclusive(RUN/'prepare.lock'):
        manifest=RUN/'input_manifest.json'
        # 步骤1：恢复不覆盖已有输入或改变人工干预，不调用任何模型。
        if manifest.exists():
            saved=read_json(manifest)
            for p,h in saved['sources'].items():
                if sha256(p)!=h:raise ValueError('Diagnostic source changed')
            for rel,h in saved['files'].items():
                if sha256(RUN/rel)!=h:raise ValueError('Frozen diagnostic input changed')
            print('Prepared input recovery passed; forwards 0',flush=True);return
        processor=AutoProcessor.from_pretrained(QWEN_PATH,local_files_only=True)
        ip=processor.image_processor
        if (ip.patch_size,ip.temporal_patch_size,ip.merge_size)!=(14,2,2):raise ValueError('Processor geometry changed')
        sources={str(Path(__file__)):sha256(__file__)};files={};observations={}
        for q,change in CHANGES.items():
            history=PARENT/'imported/results/RD-1.2-exp12'/(q+'.json');pp=BASE/'pools'/(q+'.json')
            sources.update({str(p):sha256(p) for p in [history,pp]})
            old,pool=read_json(history),read_json(pp);frames=old['selection']['selected_frames']
            add=[pool['candidates'][i] for i in change['add']]
            removed={pool['candidates'][i]['source_pts'] for i in change['remove']}
            if len(frames)!=32 or not len(add)==len(removed)==2:raise ValueError('Intervention budget differs')
            if any(i>=pool['initial_count'] for i in change['add']+change['remove']):raise ValueError('Expected coarse identities')
            if not removed<={f['source_pts'] for f in frames}:raise ValueError('Removal not in old input')
            if {f['source_pts'] for f in add}&{f['source_pts'] for f in frames}:raise ValueError('Addition already selected')
            repaired=sorted([f for f in frames if f['source_pts'] not in removed]+add,key=lambda f:f['source_pts'])
            observations[q]=dict(historical_result=str(history),historical_gpu=old['gpu'],
                                  replacements=change,conditions={})
            # 步骤2：每次仅替换两帧，其他30帧不动，按真实时间恢复顺序。
            for name,selected in [('original',frames),('repair',repaired)]:
                selection=dict(old['selection'],selected_frames=selected);validate_frames(selection)
                if len(selected)!=32:raise ValueError('Final count not 32')
                images,hashes=FrameProvider(selection).decode()
                if name=='original' and hashes!=old['rgb_sha256']:raise ValueError('Historical RGB mismatch')
                try:
                    data=ip(images=None,videos=[np.stack([np.asarray(im.convert('RGB')) for im in images])],
                            size=dict(shortest_edge=100352,longest_edge=602112),return_tensors='pt')
                finally:
                    for im in images:im.close()
                digest=tensor_hash(data['pixel_values_videos']);grid=data['video_grid_thw'].tolist()
                if grid!=old['answer']['observed']['video_grid_thw']:raise ValueError('Video token grid changed')
                if name=='original' and digest!=old['answer']['observed']['processor_pixel_sha256']:
                    raise ValueError('Historical processor tensor mismatch')
                observations[q]['conditions'][name]=dict(processor_pixel_sha256=digest,video_grid_thw=grid)
                rel=f'selections/{name}/{q}.json'
                durable(RUN/rel,dict(question_id=q,method=name,selection=selection,rgb_sha256=hashes,
                                    scope='human-directed context intervention, not algorithm'))
                files[rel]=sha256(RUN/rel)
                # 步骤3：从实际模型Tensor逆排新增帧，只供事前可见性审核，不改变输入。
                if name=='repair':
                    t,h,w=grid[0]
                    patches=data['pixel_values_videos'].numpy().reshape(t,h//2,w//2,2,2,3,2,14,14)
                    pixels=patches.transpose(0,6,5,1,3,7,2,4,8).reshape(t*2,3,h*14,w*14)
                    for i,f in enumerate(selected):
                        if f['source_pts'] not in {v['source_pts'] for v in add}:continue
                        rgb=np.clip((pixels[i].transpose(1,2,0)*np.asarray(ip.image_std)+np.asarray(ip.image_mean))*255,0,255).round().astype(np.uint8)
                        rel=f'visibility/{q}_{f["source_pts"]}.png';dest=RUN/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(rgb).save(dest);files[rel]=sha256(dest)
        # 步骤4：冻结4次诊断生成上限，原输入复测不一致时暂停归因，不重新回答。
        durable(manifest,dict(status='prepared_pending_visibility_review',question_ids=list(CHANGES),
            conditions=['original','repair'],method_reference='RD-1.2-exp12',max_real_generations=4,
            frame_budget=32,physical_gpu='1',sources=sources,files=files,observations=observations,
            scope='Combined add/remove intervention; no algorithm accuracy or isolated component claim'))
        print('Four inputs frozen; visibility review pending; model calls 0',flush=True)


if __name__=='__main__':prepare()
