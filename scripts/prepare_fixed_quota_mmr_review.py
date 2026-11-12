"""有界复核材料：先显示问题和画面，结果单列；不进行模型推理。"""
import argparse
import html
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
FIXED=(('llava8','128-1'),('llava8','306-2'),('qwen32','383-2'),
       ('qwen32','597-1'),('qwen32','691-3'),('qwen32','705-1'))


def prepare(root,output,include_extra=False):
    """接口：固定六题及正式冻结的最多两题，复用旧父图板，核对来源后生成H图板。"""
    from videoqa_audit import fixed_quota_local_mmr as m
    from videoqa_audit import fixed_quota_local_mmr_run as s
    from videoqa_audit.region_scope_review import draw_board
    rows={r['question_id']:r for r in s.development_rows()};root=Path(root);output=Path(output)
    entries=[dict(profile=p,question_id=q,kind='existing_material') for p,q in FIXED]
    if include_extra:
        entries=s.read(root/'review/panel.json')['entries']
    if len(entries)>8:raise ValueError('Review panel exceeded')
    records=[];page=['<!doctype html><meta charset="utf-8"><title>固定配额局部MMR有界复核</title>',
        '<style>body{max-width:1400px;margin:auto;font:16px/1.6 sans-serif}img{max-width:100%}</style>',
        '<h1>最多8题、45分钟；已知设计案例并非盲审</h1>']
    # 步骤1问题、源图与结构先展示；方法预测仅在独立材料中记录，避免画面判断受新答案引导。
    for e in entries:
        p,q=e['profile'],e['question_id'];row=rows[q];item=dict(e,question=row['question'],options=row['options'],sources={})
        page+=['<h2>'+html.escape(p+' '+q)+'</h2><p>'+html.escape(row['question'])+'</p>',
               '<p>'+html.escape(' / '.join(row['options']))+'</p>']
        for label,mid in [('A',m.PARENT),('B',m.CONTROL)]:
            source=(s.PREVIOUS/'qa'/p/'results'/mid/(q+'.json') if label=='A' else
                    root/'selections'/p/mid/(q+'.json'))
            # 步骤2已有父图板可逐字节复用；新图板只解码冻结选择，不补采样或写候选。
            prior=s.PREVIOUS/'review'/p/q/(mid+'.jpg');prior_meta=prior.with_suffix('.json')
            dest=output/'review'/p/q/(label+'.jpg')
            if label=='A' and prior_meta.exists():
                meta=s.read(prior_meta)
                if s.sha(prior)!=meta['image_sha256'] or s.sha(source)!=meta['source_sha256']:
                    raise ValueError('Previous parent board identity changed')
                if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(prior,dest)
                if s.sha(dest)!=s.sha(prior):raise ValueError('Reused board bytes differ')
                detail=dict(source=str(source),source_sha256=s.sha(source),image_sha256=s.sha(dest),reused_previous=True)
            else:
                detail=draw_board((str(output),q,p,label,str(source)))
            rel=p+'/'+q+'/'+label+'.jpg';page.append('<p>'+label+'</p><a href="'+rel+'"><img src="'+rel+'"></a>')
            item['sources'][label]=detail
        tp=root/'traces'/p/m.CONTROL/(q+'.json');t=s.read(tp)
        item['regions']=[dict(region_id=x['region_id'],quota=x['quota'],anchor=x['anchor'],selection_order=x['selection_order']) for x in t['regions']]
        item['trace_sha256']=s.sha(tp);records.append(item)
    # 步骤3首次六题和最终最多八题分别冻结，避免后来追加槽位改写首阶段记录。
    suffix='final' if include_extra else 'fixed'
    record=dict(entries=records,maximum_questions=8,maximum_minutes=45,new_model_calls=0,labels=dict(A=m.PARENT,B=m.CONTROL),
        limitation='Known-case posthoc AI review; thumbnails are not the official model view',script_sha256=s.sha(Path(__file__)))
    s.freeze(output/('materials_'+suffix+'.json'),record)
    pagepath=output/'review'/('index_'+suffix+'.html');content='\n'.join(page)
    if not pagepath.exists():pagepath.write_text(content)
    elif pagepath.read_text()!=content:raise ValueError('Review page changed')
    return dict(status='prepared',questions=len(entries),new_model_calls=0)


def main():
    """接口：只接受本轮独立诊断与分析目录，正式结果不进入准备函数输入。"""
    import json
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--include-extra',action='store_true');a=parser.parse_args()
    root,output=Path(a.run_dir).resolve(),Path(a.output_dir).resolve()
    if not root.is_relative_to((ROOT/'outputs/diagnostics').resolve()) or not output.is_relative_to((ROOT/'outputs/analysis').resolve()):
        parser.error('Unexpected diagnostic or analysis path')
    print(json.dumps(prepare(root,output,a.include_extra),ensure_ascii=False))


if __name__=='__main__':main()
