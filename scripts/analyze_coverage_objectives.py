"""两个全量工作点的零模型覆盖目标审计与固定面板生成。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse
import html
from collections import Counter
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable,exclusive,utc
from videoqa_audit.coverage_objective_review import compare_allocations,frame_identity,trace_witness_path,region_correspondence

BASE=ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
Q07=ROOT/'outputs/full_videoqa/rd12_exp07__qwen32__videomme2700__20260930_r1'
Q12=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
L8=ROOT/'outputs/full_videoqa/llava8__five_methods__videomme2700__20261005_r1'
GEN=ROOT/'outputs/full_videoqa/exp12_generalization__videomme2700__20261005_r1'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
DEV=ROOT/'data/manifests/video_mme_development_200_20260920.json'
METHODS=('BASE-Uniform','BASE-BLIP-TopK','RD-1.2-exp07','RD-1.2-exp12')


class Reader:
    """接口：只读来源并保存字节指纹，同一路径只哈希一次，完成审计再次核验。"""
    def __init__(self):self.sources={}
    def bind(self,path):
        path=Path(path)
        if str(path) not in self.sources:self.sources[str(path)]=sha256(path)
        return path
    def read(self,path):return read_json(self.bind(path))


def result_path(root,method,q):
    """接口：兼容已冻结的开发导入目录，唯一选择存在来源，不重新生成。"""
    paths=[Path(root)/'results'/method/(q+'.json'),Path(root)/'imported/results'/method/(q+'.json')]
    matches=[p for p in paths if p.exists()]
    if len(matches)!=1:raise ValueError('Result source is missing or ambiguous: '+q)
    return matches[0]


def load_pair(profile,q,reader):
    """接口：读取同配置池、前缀和两覆盖目标，M8特征映射回原M32资产。"""
    if profile=='qwen32':
        pool=reader.read(BASE/'pools'/(q+'.json'))
        a=reader.read(Q07/'traces'/(q+'.json'));b=reader.read(Q12/'traces'/(q+'.json'))
        for method,root in [(METHODS[2],Q07),(METHODS[3],Q12)]:
            item=reader.read(root/'selections'/method/(q+'.json'))
            for p,h in item['sources'].items():
                if sha256(reader.bind(p))!=h:raise ValueError('Frozen selection origin changed')
    else:
        pool=reader.read(L8/'inputs/pools'/(q+'.json'));t=reader.read(L8/'inputs/traces'/(q+'.json'))
        a,b=t['exp07'],t['exp12']
        if a!=b['prefix_trace']:raise ValueError('M8 parent prefix differs')
        check=reader.read(L8/'inputs/checkpoints'/(q+'.json'))
        for p,h in check['sources'].items():
            if sha256(reader.bind(p))!=h:raise ValueError('M8 frozen origin changed')
    reader.bind(BASE/'features'/(q+'.npy'))
    # 步骤1：细帧归属从等权记录补到加权展示，不改原轨迹。
    b=dict(b,regions=[dict(y,members=x['members']) for x,y in zip(a['regions'],b['regions'])])
    return pool,a,b


def answers(profile,q,reader):
    """接口：五个只读结果源只用于事后统计和面板冻结，不传给选择函数。"""
    roots={METHODS[0]:BASE,METHODS[1]:BASE,METHODS[2]:Q07,METHODS[3]:Q12} if profile=='qwen32' else {m:L8 for m in METHODS}
    if profile=='qwen32':roots['REF-WFS-SB']=GEN/'wfs32'
    return {m:dict(path=str(result_path(p,m,q)),value=reader.read(result_path(p,m,q))) for m,p in roots.items()}


def freeze_panel(data,records,development_videos):
    """接口：固定两诊断+十槽位，按已确认次序及开发视频外优先，不因审核结论换题。"""
    byid={r['question_id']:r for r in data};panel=[];used=set();slots=[]
    def add(q,slot,profiles):
        panel.append(dict(question_id=q,slot=slot,profiles=profiles,known_posthoc=q in ['409-3','556-1']))
        used.add(byid[q]['video_id']);slots.append(dict(slot=slot,question_id=q))
    for q in ['409-3','556-1']:add(q,'known_'+q,['qwen32'])
    ordered=sorted(data,key=lambda r:(r['video_id'] in development_videos,data.index(r)))
    def take(slot,profiles,predicate):
        q=next((r['question_id'] for r in ordered if r['question_id'] not in ['491-2','321-2']
            and r['video_id'] not in used and predicate(r['question_id'])),None)
        if q is None:slots.append(dict(slot=slot,question_id=None));return
        add(q,slot,profiles)
    # 步骤1：四个各自得益槽位，排序反转另外两槽，已用视频不重复。
    for profile in ['qwen32','llava8']:
        for pattern in ['left_only','right_only']:
            take(profile+'_'+pattern,[profile],lambda q,p=profile,k=pattern:records[p][q]['correctness_cell']==k)
    take('reversal_q12_l07',['qwen32','llava8'],lambda q:records['qwen32'][q]['correctness_cell']=='right_only' and records['llava8'][q]['correctness_cell']=='left_only')
    take('reversal_q07_l12',['qwen32','llava8'],lambda q:records['qwen32'][q]['correctness_cell']=='left_only' and records['llava8'][q]['correctness_cell']=='right_only')
    for profile,reference in [('qwen32','REF-WFS-SB'),('llava8','BASE-Uniform')]:
        take(profile+'_shared_failure',[profile],lambda q,p=profile,m=reference:records[p][q]['correctness_cell']=='both_wrong' and records[p][q]['answers'][m]['correct'])
    for profile in ['qwen32','llava8']:
        take(profile+'_protection',[profile],lambda q,p=profile:records[p][q]['correctness_cell']=='both_correct' and records[p][q]['answers']['BASE-Uniform']['correct'] and not records[p][q]['same_input'])
    if len(panel)>12 or len({p['question_id'] for p in panel})!=len(panel):raise ValueError('Panel bound violated')
    return dict(questions=panel,slots=slots,rule='two fixed diagnostics; outside-development-video first; frozen manifest order; unique videos; no slot substitution')


def analyze(root):
    """接口：全5400同配置记录、75事实、跨预算对应及固定面板；所有模型调用为0。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);reader=Reader();data=rows()
    with exclusive(root/'analysis.lock'):
        if (root/'summary.json').exists():
            saved=read_json(root/'summary.json')
            for p,h in saved['sources'].items():
                if sha256(p)!=h:raise ValueError('Resume source changed')
            return saved
        reader.bind(Path(__file__));reader.bind(ROOT/'src/videoqa_audit/coverage_objective_review.py')
        manifest=ROOT/'data/manifests/video_mme_full_2700.json';reader.bind(manifest)
        dev=reader.read(DEV)['rows'];dv={r['video_id'] for r in dev}
        specifications={p:reader.read(r/'protocol.json') for p,r in [('q07',Q07),('q12',Q12),('llava8',L8),('wfs',GEN/'wfs32')]}
        for folder in [Q07,Q12,L8,GEN/'wfs32']:
            if reader.read(folder/'completion_audit.json')['status']!='passed':raise ValueError('Source full audit incomplete')
        records={p:{} for p in ['qwen32','llava8']};summaries={};started=time.perf_counter()
        facts=reader.read(FACTS)['entries']
        if len(facts)!=75 or len({f['question_id'] for f in facts})!=42:raise ValueError('Proxy denominator changed')
        witnesses={};cross=[]
        # 步骤1：900视频按实际字节核验，防止后续图板悄悄使用另一原片。
        source=specifications['q07']
        for vid,st in source['video_stats'].items():
            if sha256(reader.bind(st['path']))!=source['video_hashes'][vid]:raise ValueError('Video content changed')
        for row in data:
            q=row['question_id'];pairs={}
            for profile in ['qwen32','llava8']:
                pool,a,b=load_pair(profile,q,reader);pair=compare_allocations(pool,a,b);pairs[profile]=(pool,a,b)
                raw=answers(profile,q,reader);values={};sources={}
                for m,v in raw.items():
                    r=v['value'];answer=r['answer'];o=answer['observed']
                    if r['question']!=row['question'] or r['options']!=row['options'] or r['correct']!=(answer['parsed_answer']=='ABCD'[row['answer_index']]):
                        raise ValueError('Question/scoring changed')
                    if len(r['selection']['selected_frames'])!=len(r['rgb_sha256']):raise ValueError('RGB population differs')
                    values[m]=dict(correct=r['correct'],prediction=answer['parsed_answer'],raw_output=answer['raw_output'],
                        shared=r.get('shared',False),representative=r.get('representative',m),imported=r.get('imported',False))
                    sources[m]=v['path']
                left,right=raw[METHODS[2]]['value'],raw[METHODS[3]]['value']
                for m,r,ids in [(METHODS[2],left,pair['selected_left']),(METHODS[3],right,pair['selected_right'])]:
                    expected={frame_identity(pool['candidates'][i]) for i in ids}
                    if expected!={frame_identity(f) for f in r['selection']['selected_frames']}:raise ValueError('Actual source differs from prefix')
                oa,ob=left['answer']['observed'],right['answer']['observed']
                keys=['input_token_ids','model_generation_config','source_frames']
                keys+=['processed_size','generation_kwargs','second_per_grid_ts'] if profile=='qwen32' else ['max_new_tokens']
                if any(oa[k]!=ob[k] for k in keys):raise ValueError('Paired QA protocol differs')
                la,lb=values[METHODS[2]],values[METHODS[3]]
                cell='both_correct' if la['correct'] and lb['correct'] else 'left_only' if la['correct'] else 'right_only' if lb['correct'] else 'both_wrong'
                same=[frame_identity(f) for f in left['selection']['selected_frames']]==[frame_identity(f) for f in right['selection']['selected_frames']]
                if same and left['rgb_sha256']!=right['rgb_sha256']:raise ValueError('Same source RGB differs')
                prediction_changed=la['prediction']!=lb['prediction']
                # 四类可复算标记：对错改变属于预测改变的子集，分别保存不伪造互斥。
                record=dict(question_id=q,video_id=row['video_id'],stratum=row['stratum'],task_type=row.get('task_type','unknown'),
                    outside_development_videos=row['video_id'] not in dv,answers=values,answer_sources=sources,
                    correctness_cell=cell,same_input=same,different_input_same_prediction=not same and not prediction_changed,
                    prediction_changed=prediction_changed,correctness_changed=la['correct']!=lb['correct'],allocation=pair)
                records[profile][q]=record
                # 步骤2：主代理逐见证映射精确源身份，不把当前答案反向写入事实目录。
                for fact in [f for f in facts if f['question_id']==q]:
                    lookup={c['source_pts']:frame_identity(c) for c in pool['candidates'][:pool['initial_count']]}
                    keys=[lookup[p] for p in fact['witness_pts']]
                    witnesses.setdefault(profile,[]).append(dict(question_id=q,fact_id=fact['fact_id'],description=fact['description'],role=fact['role'],
                        exp07=trace_witness_path(pool,a,keys),exp12=trace_witness_path(pool,b,keys)))
                if q in ['409-3','556-1']:
                    initial=[77,80] if q=='409-3' else [42]
                    keys=[frame_identity(pool['candidates'][i]) for i in initial]
                    witnesses.setdefault(profile+'_supplemental',[]).append(dict(question_id=q,
                        exp07=trace_witness_path(pool,a,keys),exp12=trace_witness_path(pool,b,keys)))
                durable(root/'records'/profile/(q+'.json'),record)
            cross.append(dict(question_id=q,video_id=row['video_id'],
                regions=region_correspondence(pairs['llava8'][0],pairs['llava8'][1],pairs['qwen32'][0],pairs['qwen32'][1]),
                limitation='M8/K8 versus M32/K32 changes model and upstream candidate scope; descriptive only'))
            if len(cross)%100==0:durable(root/'status.json',dict(status='analyzing',completed=len(cross),total=2700,seconds=time.perf_counter()-started,utc=utc()))
        # 步骤3：全量分母、结构分组和事实保持独立；面板按预定规则冻结。
        for profile,group in records.items():
            summaries[profile]={}
            for name,subset in [('all2700',list(group.values()))]+[(x,[r for r in group.values() if r['stratum']==x]) for x in ['short','medium','long']]+[
                    ('task__'+x,[r for r in group.values() if r['task_type']==x]) for x in sorted({r['task_type'] for r in group.values()})]:
                summaries[profile][name]=dict(n=len(subset),correctness_cells=dict(Counter(r['correctness_cell'] for r in subset)),
                    question_ids={cell:[r['question_id'] for r in subset if r['correctness_cell']==cell] for cell in ['both_correct','left_only','right_only','both_wrong']},
                    same_input=sum(r['same_input'] for r in subset),different_input_same_prediction=sum(r['different_input_same_prediction'] for r in subset),
                    prediction_changed=sum(r['prediction_changed'] for r in subset),correctness_changed=sum(r['correctness_changed'] for r in subset),
                    transferred_slots=dict(Counter(r['allocation']['transferred_slots'] for r in subset)),
                    shared_fail_uniform_correct=[r['question_id'] for r in subset if r['correctness_cell']=='both_wrong' and r['answers']['BASE-Uniform']['correct']],
                    shared_fail_wfs_correct=[r['question_id'] for r in subset if r['correctness_cell']=='both_wrong' and r['answers'].get('REF-WFS-SB',{}).get('correct')])
        panel=freeze_panel(data,records,dv);durable(root/'panel.json',panel)
        durable(root/'witness_paths.json',witnesses);durable(root/'cross_configuration.json',cross)
        summary=dict(status='completed',profiles=summaries,facts_denominator=dict(questions=42,facts=75),
            facts={p:dict(exp07_hits=sum(r['exp07']['exact_hit'] for r in witnesses[p]),exp12_hits=sum(r['exp12']['exact_hit'] for r in witnesses[p]),
                exp12_miss_quota_reachable=sum(not r['exp12']['exact_hit'] and r['exp12']['any_quota_reachable'] for r in witnesses[p])) for p in records},
            panel=panel,new_model_calls=0,sources=reader.sources,seconds=time.perf_counter()-started,utc=utc())
        durable(root/'summary.json',summary);print({p:{k:v for k,v in summaries[p]['all2700'].items() if k not in ['question_ids','shared_fail_uniform_correct','shared_fail_wfs_correct']} for p in records},flush=True)
        print('Panel',[(x['slot'],x['question_id']) for x in panel['questions']],flush=True)
        return summary


def main():
    """接口：只读分析入口，输出独立目录；完成恢复核验同一来源，不重新评分。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['analyze','render_qwen','render_llava','audit'],default='analyze');args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid audit ID')
    root=ROOT/'outputs/analysis'/args.run_id
    if args.stage=='analyze':analyze(root)
    elif args.stage=='audit':audit(root)
    else:render(root,'qwen32' if args.stage=='render_qwen' else 'llava8')


def image_board(images,frames,path):
    """接口：原画面等比图板，序号与PTS可回到真实源帧，不制造示意画面。"""
    from PIL import Image,ImageDraw
    width=1280;height=230*((len(images)+3)//4)
    result=Image.new('RGB',(width,max(230,height)),'white');draw=ImageDraw.Draw(result)
    for i,(im,frame) in enumerate(zip(images,frames)):
        view=im.copy();view.thumbnail((316,204));x=(i%4)*320;y=(i//4)*230
        result.paste(view,(x,y));draw.text((x,y+205),f"{i+1} | {frame['timestamp_seconds']:.3f}s | src {frame['source_frame_index']}",fill='black');view.close()
    result.save(path,quality=95);result.close()


def render(root,profile):
    """接口：按冻结12题生成真实原图、模型尺寸视图、差异与原始得分追踪，CPU处理0前向。"""
    import numpy as np
    import torch
    from PIL import Image
    from videoqa_budget32.frames import FrameProvider
    from videoqa_audit.coverage_objective_review import component_at
    from videoqa_budget32.observe import tensor_hash
    root=Path(root);summary=read_json(root/'summary.json');panel=read_json(root/'panel.json')
    data={r['question_id']:r for r in rows()};reader=Reader();torch.set_num_threads(2)
    if profile=='qwen32':
        from transformers import AutoProcessor
        from videoqa_full.state import QWEN_PATH
        from videoqa_full.qwen_backend import MIN_PIXELS,MAX_PIXELS
        processor=AutoProcessor.from_pretrained(QWEN_PATH,local_files_only=True,use_fast=False,
            min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS,size={'shortest_edge':MIN_PIXELS,'longest_edge':MAX_PIXELS})
    else:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
        processor=SigLipImageProcessor()
    for case in panel['questions']:
        if profile not in case['profiles']:continue
        q=case['question_id'];record=read_json(root/'records'/profile/(q+'.json'));dest=root/'panel'/q/profile;dest.mkdir(parents=True,exist_ok=True)
        pool,a,b=load_pair(profile,q,reader);features=np.load(BASE/'features'/(q+'.npy'),allow_pickle=False)
        if profile=='llava8':features=features[pool['original_candidate_indices']]
        manifest={};banks={}
        # 步骤1：精确读取每种实际输入，RGB及官方处理器字节对齐已保存观测。
        for ordinal,method in enumerate([METHODS[0],METHODS[2],METHODS[3]]):
            path=record['answer_sources'][method]
            if sha256(path)!=summary['sources'][path]:raise ValueError('Panel answer source changed')
            item=reader.read(path);images,hashes=FrameProvider(item['selection']).decode()
            if hashes!=item['rgb_sha256']:raise ValueError('Panel actual RGB changed')
            observed=item['answer']['observed'];times=item['selection']['selected_frames'];banks[method]=(images,times)
            if profile=='qwen32':
                context=observed['context'];messages=[dict(role='system',content='You are a helpful assistant.'),
                    dict(role='user',content=[dict(type='video'),dict(type='text',text=context)])]
                prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                processed=processor(text=[prompt],videos=[np.stack([np.asarray(im) for im in images])],fps=2.0,
                    images_kwargs={'size':{'shortest_edge':MIN_PIXELS,'longest_edge':MAX_PIXELS}},padding=False,truncation=False,return_tensors='pt')
                digest=tensor_hash(processed['pixel_values_videos'])
                if digest!=observed['model_pixel_sha256'] or processed['input_ids'][0].tolist()!=observed['input_token_ids']:
                    raise ValueError('Qwen CPU processor identity changed')
                h,w=observed['processed_size'];model_images=[im.resize((w,h),Image.Resampling.BICUBIC) for im in images]
            else:
                pixels=processor.preprocess(images,return_tensors='pt')['pixel_values'].to(torch.bfloat16)
                digest=tensor_hash(pixels)
                if digest!=observed['pixel_sha256']:raise ValueError('LLaVA CPU processor identity changed')
                model_images=[Image.fromarray(np.clip(np.rint((v.float().permute(1,2,0).numpy()+1)*127.5),0,255).astype(np.uint8)) for v in pixels]
            image_board(images,times,dest/(str(ordinal)+'_raw.jpg'))
            image_board(model_images,times,dest/(str(ordinal)+'_model.jpg'))
            for i,(im,model_image) in enumerate(zip(images,model_images)):
                im.save(dest/f'{ordinal}_{i:02d}_raw.png');model_image.save(dest/f'{ordinal}_{i:02d}_model.png');model_image.close()
            manifest[method]=dict(anonymous_group=ordinal,answer_source=path,source_sha256=sha256(path),
                actual_processor_sha256=digest,frames=times,raw_board=f'{ordinal}_raw.jpg',model_board=f'{ordinal}_model.jpg')
        # 步骤2：只展示变化帧与局部次序，不把得分较高直接标为有效证据。
        for side,method in [('left',METHODS[2]),('right',METHODS[3])]:
            target={frame_identity(x['frame']) for x in record['allocation'][side+'_only']}
            images,times=banks[method];pairs=[(im,f) for im,f in zip(images,times) if frame_identity(f) in target]
            image_board([x[0] for x in pairs],[x[1] for x in pairs],dest/(side+'_changes.jpg'))
        details=[]
        for ra,rb in zip(a['regions'],b['regions']):
            selected=[];steps=[]
            for i in ra['prefix']:
                component=component_at(pool,features,ra,i,selected)
                steps.append(dict(frame=pool['candidates'][i],**component));selected.append(i)
            details.append(dict(region_id=ra['region_id'],seed_indices=next(r['seed_indices'] for r in pool['plan']['regions'] if r['region_id']==ra['region_id']),
                quota07=a['region_budgets'][str(ra['region_id'])],quota12=b['region_budgets'][str(ra['region_id'])],prefix_steps=steps))
        durable(dest/'manifest.json',dict(inputs=manifest,regions=details,plan=pool['plan'],
            semantic_review='pending',new_model_calls=0,sources=reader.sources))
        for images,_ in banks.values():
            for im in images:im.close()
        row=data[q];page=['<!doctype html><meta charset="utf-8"><title>'+q+'</title>',
            '<style>body{max-width:1400px;margin:auto;font:16px sans-serif}img{max-width:100%}.names{display:none}</style>',
            '<h1>'+html.escape(q)+' / '+profile+'</h1><p>'+html.escape(row['question'])+'</p><p>'+html.escape(' / '.join(row['options']))+'</p>',
            '<p>先查看问题与画面；已有诊断明确属于事后审核。模型尺寸视图经官方处理器字节核验，画面可读不保证模型理解。</p>',
            '<button onclick="document.querySelectorAll(\'.names\').forEach(x=>x.style.display=\'block\')">展开方法与回答</button>',
            '<h2>两组变化帧</h2><img src="left_changes.jpg"><img src="right_changes.jpg">']
        for method,item in manifest.items():
            i=item['anonymous_group'];answer=record['answers'][method]
            page += [f'<h2>输入{i+1}</h2>',f'<div class="names">{method} / '+html.escape(answer['raw_output'])+' / '+str(answer['correct'])+'</div>',
                f'<img src="{i}_raw.jpg"><details><summary>模型尺寸图板及原帧链接</summary><img src="{i}_model.jpg">',
                ' '.join(f'<a href="{i}_{j:02d}_raw.png">原帧{j+1}</a> <a href="{i}_{j:02d}_model.png">输入视图</a>' for j in range(len(item['frames'])))+'</details>']
        page+=['<details><summary>区域、种子、配额及局部前缀得分</summary><pre>'+html.escape(__import__('json').dumps(details,ensure_ascii=False,indent=2))+'</pre></details>']
        (dest/'index.html').write_text('\n'.join(page),encoding='utf-8')
    links=['<!doctype html><meta charset="utf-8"><h1>覆盖目标固定面板</h1><p>零模型调用；语义初审及争议另见review.json。相同输入引用不是重复试验。</p>']
    for case in panel['questions']:
        links.append('<p>'+case['slot']+' / '+case['question_id']+' '+ ' '.join(f'<a href="panel/{case["question_id"]}/{p}/index.html">{p}</a>' for p in case['profiles'])+'</p>')
    (root/'index.html').write_text('\n'.join(links),encoding='utf-8')
    durable(root/(profile+'_render.json'),dict(status='completed',processor_only=True,new_model_calls=0,utc=utc()))


def audit(root):
    """接口：再次核验全部来源与5400统计记录，完成恢复不改历史数据或新增调用。"""
    root=Path(root);summary=read_json(root/'summary.json')
    for p,h in summary['sources'].items():
        if sha256(p)!=h:raise ValueError('Historical source changed: '+p)
    for profile in ['qwen32','llava8']:
        if len(list((root/'records'/profile).glob('*.json')))!=2700:raise ValueError('Missing profile record')
        if read_json(root/(profile+'_render.json'))['status']!='completed':raise ValueError('Panel assets incomplete')
    durable(root/'completion_audit.json',dict(status='passed',historical_sources_unchanged=len(summary['sources']),
        questions_per_profile=2700,new_model_calls=0,utc=utc()))


if __name__=='__main__':main()
