"""固定12题的精确源图与局部竞争复核；只解码既有帧，不运行模型。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
import html
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_budget32.frames import FrameProvider
from videoqa_audit.cross_dataset_review import METHODS
from videoqa_audit.coverage_objective_review import component_at
from analyze_cross_dataset_review import ML, pool_pair, Reader, publish


def draw(images, frames, path, labels=None, columns=4):
    """接口：原RGB等比图板，标题保留源时间/编号；不把缩略图称为模型实际输入。"""
    from PIL import Image, ImageDraw
    canvas=Image.new('RGB',(columns*360,235*((len(images)+columns-1)//columns)),'white')
    painter=ImageDraw.Draw(canvas)
    # 步骤1：图板只复制源图缩放，不改返回给审核者的原图或历史问答像素。
    for i,(image,frame) in enumerate(zip(images,frames)):
        thumb=image.copy();thumb.thumbnail((356,198))
        x,y=(i%columns)*360,(i//columns)*235
        canvas.paste(thumb,(x,y));thumb.close()
        painter.text((x+2,y+199),f"{i+1}: {frame['timestamp_seconds']:.3f}s src {frame['source_frame_index']}",fill='black')
        if labels:painter.text((x+2,y+215),str(labels[i])[:57],fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=95);canvas.close()


def render(root):
    """接口：固定清单生成最多12题图板和精确竞争分量；实际语义初审另存，禁止自动补标签。"""
    root=Path(root);panel=read_json(root/'panel.json');reader=Reader();reader.bind(Path(__file__))
    if len(panel['questions'])>12:raise ValueError('Panel limit exceeded')
    vbank={r['question_id']:r for r in read_json(ROOT/'data/manifests/video_mme_full_2700.json')['rows']}
    mbank={r['question_id']:r for r in read_json(ML/'manifest.json')['rows']}
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>双数据集选择机制复核</title>',
          '<style>body{max-width:1500px;margin:auto;padding:24px;font:16px/1.65 sans-serif}img{max-width:100%}details{border:1px solid #ccc;padding:8px;margin:10px 0}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}textarea{width:95%}</style>',
          '<h1>双数据集：检索帧保留与覆盖替代复核</h1>',
          '<p>事后审核，模型调用0。先看题目/图板再展开方法与回答；并非完全盲审。原RGB图板可点击放大，缺少旧见证不自动等于缺少事实。留言仅保存在浏览器，导出后才归档。</p>']
    exports={};cases=[];start=time.perf_counter()
    # 步骤1：只读取已冻结面板，不因图像解释或答案变化换题。
    for case in panel['questions']:
        dataset,q=case['dataset'],case['question_id'];row=(mbank if dataset=='mlvu' else vbank)[q]
        folder=root/'review'/dataset/q
        page += [f'<h2>{html.escape(dataset+" / "+q)}</h2><p>'+html.escape(row['question'])+'</p>',
                 '<p>'+html.escape(' / '.join(f'{c}: {o}' for c,o in zip('ABCD',row['options'])))+'</p>']
        details={}
        for profile in case['profiles']:
            record=read_json(root/'records'/dataset/profile/(q+'.json'))
            pool,a,b,pp,fp=pool_pair(dataset,profile,q,reader)
            reader.bind(fp)
            raw=np.load(fp,mmap_mode='r',allow_pickle=False)
            features=raw[pool['original_candidate_indices']] if dataset=='videomme' and profile=='llava8' else raw
            if len(features)!=len(pool['candidates']):raise ValueError('Feature/source mapping differs')
            selected={name:record['structure']['selected_indices'][name] for name in ('exp07','exp12')}
            regions={r['region_id']:r for r in a['regions']};components=[]
            # 步骤2：实际同区代表与首次竞争赢家均保存，视觉相似只作为需审核的替代代理。
            for path in record['structure']['topk_paths']:
                i=path['candidate_index'];r=regions[path['region_id']]
                entry=dict(path)
                if len(r['prefix'])>1:
                    before=r['prefix'][:1]
                    entry['first_target_score']=component_at(pool,features,r,i,before)
                    entry['first_actual_winner']=component_at(pool,features,r,r['prefix'][1],before)
                for name in ('exp07','exp12'):
                    ids=r['prefix'][:(a if name=='exp07' else b)['region_budgets'][str(r['region_id'])]]
                    sims=np.clip(features[ids]@features[i],0,1)
                    j=int(np.argmax(sims));other=ids[j]
                    entry[name]=dict(path[name],visual_representative=other,
                        representative_seconds=pool['candidates'][other]['timestamp_seconds'],
                        cosine=float(sims[j]),representative_is_same_source=i==other)
                components.append(entry)
            publish(folder/profile/'competition.json',dict(paths=components,
                limitation='Highest cosine representative is not established semantic equivalence',new_model_calls=0))
            profile_detail={};page += ['<h3>'+profile+'</h3>']
            for index,method in enumerate(METHODS,1):
                source=record['answer_sources'][method];item=reader.read(source);frames=item['selection']['selected_frames']
                images,rgb=FrameProvider(item['selection']).decode()
                if rgb!=item['rgb_sha256']:raise ValueError('Review RGB differs from historical QA')
                out=folder/profile/f'input_{index}.jpg'
                draw(images,frames,out)
                for im in images:im.close()
                relative=str(out.relative_to(root/'review'))
                meta=dict(source=source,source_sha256=sha256(source),image_sha256=sha256(out),
                          frames=frames,rgb_sha256=rgb,method=method,source_view='original_RGB_thumbnail')
                publish(out.with_suffix('.json'),meta)
                exports[str(out.relative_to(root))]=sha256(out)
                profile_detail[method]=meta
                answer=record['answers'][method]
                page += [f'<details><summary>输入组{index}：{len(frames)}帧</summary>',
                    f'<a href="{relative}"><img loading="lazy" src="{relative}"></a>',
                    '<details><summary>方法与原始回答</summary><p>'+html.escape(method+' / '+answer['raw_output'])+
                    f" / native={answer['native_correct']} / normalized={answer['correct']}</p></details></details>"]
            # 步骤3：候选与近似替代并排，最多展示一个Top-K集合，不扩为全片逐帧审计。
            removed=[x for x in components if x['exp07']['state']!='selected' or x['exp12']['state']!='selected']
            if removed:
                ids=[];labels=[]
                for e in removed:
                    for i,label in [(e['candidate_index'],f"TopK rank {e['rank']} ITM {e['itm']:.4f}"),
                                    (e['exp07']['visual_representative'],f"07 {e['exp07']['state']} cos {e['exp07']['cosine']:.4f}"),
                                    (e['exp12']['visual_representative'],f"12 {e['exp12']['state']} cos {e['exp12']['cosine']:.4f}")]:
                        ids.append(i);labels.append(label)
                unique=sorted(set(ids),key=lambda i:pool['candidates'][i]['source_pts'])
                # 审核拼图可能超过模型预算，分批读取既有源帧；不放宽正式输入接口。
                images=[]
                for start in range(0,len(unique),32):
                    batch,_=FrameProvider(dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in unique[start:start+32]])).decode()
                    images.extend(batch)
                lookup=dict(zip(unique,images));out=folder/profile/'retrieval_replacements.jpg'
                draw([lookup[i] for i in ids],[pool['candidates'][i] for i in ids],out,labels,columns=3)
                for im in images:im.close()
                relative=str(out.relative_to(root/'review'));exports[str(out.relative_to(root))]=sha256(out)
                page += ['<details open><summary>未保留Top-K与各版最高相似代表（逐行三列）</summary>',
                         f'<a href="{relative}"><img loading="lazy" src="{relative}"></a></details>']
            comp=str((folder/profile/'competition.json').relative_to(root/'review'))
            page += [f'<p><a href="{comp}">前缀位置、配额、L/D与实际竞争帧</a></p>']
            details[profile]=profile_detail
        key=dataset+'/'+q
        page += ['<details><summary>标准答案及槽位</summary>'+html.escape('ABCD'[row['answer_index']]+' / '+case['slot'])+'</details>',
                 f'<textarea rows="3" data-review-key="{key}" placeholder="必要事实、替代见证与争议"></textarea>']
        cases.append(dict(case,question=row['question'],options=row['options'],answer='ABCD'[row['answer_index']],
                          boards=details,semantic_review_status='pending_actual_review'))
    page += ['<button id="export">导出用户复核</button><script>const notes=document.querySelectorAll("textarea[data-review-key]");notes.forEach(t=>{const k="dual_review/"+t.dataset.reviewKey;t.value=localStorage.getItem(k)||"";t.oninput=()=>localStorage.setItem(k,t.value)});document.getElementById("export").onclick=()=>{const v=Object.fromEntries(Array.from(notes).map(t=>[t.dataset.reviewKey,t.value]));const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(v,null,2)],{type:"application/json"}));a.download="user_review.json";a.click()};</script></html>']
    (root/'review/index.html').write_text('\n'.join(page),encoding='utf-8')
    durable(root/'review/manifest.json',dict(cases=cases,exports=exports,sources=reader.sources,
        new_model_calls=0,semantic_review_status='pending_actual_review',
        render_seconds=time.perf_counter()-start,visual_review_seconds=0))
    print(dict(cases=len(cases),boards=len(exports),new_model_calls=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):raise ValueError('Invalid run ID')
    render(ROOT/'outputs/analysis'/args.run_id)
