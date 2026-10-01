"""帧组能力探针的事前固定设计，供两种评分器共同读取，不选择正确选项。"""
from pathlib import Path
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.framegroup_embedding import QUERY_INSTRUCTION,DOCUMENT_INSTRUCTION,MAX_PIXELS,MIN_PIXELS


def query_views(case):
    """接口：原问题与全部四个原选项对称展开；不接收/读取标准答案作分支选择。"""
    if len(case['options'])!=4:raise ValueError('Expected four original options')
    return [('question',case['question'])]+[('option_'+letter,case['question']+'\n'+option) for letter,option in zip('ABCD',case['options'])]


def create_design(run):
    """接口：冻结文本、图像、正/逆序帧组及聚合口径，真实输出出现后不得改输入配方。"""
    run=Path(run);source=run/'assets_manifest_r3.json';assets=read_json(source)
    for p,h in assets['sources'].items():assert sha256(p)==h,p
    if set(assets['cases'])!={'395-2','409-3'}:raise ValueError('Unexpected diagnostic cases')
    jobs=[];frame_sources={};queries={}
    # 步骤1：先独立编码全部查询和唯一图像，再处理正序/逆序；不生成或选择答案。
    for q,case in assets['cases'].items():
        queries[q]=dict(query_views(case))
        jobs.extend(dict(id=q+'__'+name,kind='text',question_id=q,query_variant=name,text=text) for name,text in query_views(case))
        for pts,row in sorted(case['frames'].items(),key=lambda kv:kv[1]['source_pts']):frame_sources[q+'__'+pts]=row
    jobs.extend(dict(id='image__'+fid,kind='image',frame_ids=[fid]) for fid in frame_sources)
    for q,case in assets['cases'].items():
        for name,frames in case['clips'].items():
            if len(frames)!=10 or len(set(frames))!=10:raise ValueError('Ten unique source frames required')
            ids=[q+'__'+p for p in frames]
            for order,seq in [('original',ids),('reversed',list(reversed(ids)))]:
                jobs.append(dict(id=q+'__'+name+'__'+order,kind='video',question_id=q,group=name,order=order,frame_ids=seq))
    assert len(frame_sources)==32 and len(jobs)==len({x['id'] for x in jobs})==52
    # 步骤2：处理与评价定义都冻结；逆序只测敏感性，不自动标为负例。
    design=dict(asset_manifest=str(source),asset_sha256=sha256(source),question_ids=list(assets['cases']),jobs=jobs,frame_sources=frame_sources,queries=queries,
        embedding_model='/home/models/Qwen/Qwen3-VL-Embedding-8B',query_instruction=QUERY_INSTRUCTION,document_instruction=DOCUMENT_INSTRUCTION,
        embedding_configuration=dict(dtype='bfloat16',attention='sdpa',pooling='last valid token of checkpoint template',normalization='float32 L2',
            max_input_tokens=8192,min_frame_pixels=MIN_PIXELS,max_frame_pixels=MAX_PIXELS,encoding_fps=2,source_frames_per_group=10,seed=2027),
        real_embedding_forwards=52,synthetic_embedding_forwards=3,blip_frame_text_pairs=160,blip_batches=15,blip_synthetic_batches=1,final_qa_calls=0,
        primary='Within-question comparison using original question; option views secondary and symmetric',
        aggregation=['BLIP frame max','BLIP frame mean','embedding single-frame max','normalized mean image embedding','ordered video embedding','reversed video embedding'],
        scope='Human-constructed capability probe, not an automatic method or accuracy benchmark; no answers generated')
    target=run/'design_protocol_r2.json'
    if target.exists():assert read_json(target)==design
    else:durable(target,design)
    return design
