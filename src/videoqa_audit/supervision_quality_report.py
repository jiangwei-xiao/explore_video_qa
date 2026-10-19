"""监督质量诊断的离线统计与图板；任何计分/审核结果都不反馈改变输入。"""
from collections import Counter
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from videoqa_runtime.common import read_json
from videoqa_full.state import durable
from videoqa_set_value.data import METHODS

CONDITIONS=('O','G','W','L','R')


def paired_comparison(rows, left, right):
    """接口：每题一个独立视频的配对差值，固定10000次视频重抽与完整四格题号。"""
    if not rows:return dict(questions=0,interval_95=None,difference=None)
    a=np.asarray(left,dtype=int);b=np.asarray(right,dtype=int)
    if len(a)!=len(rows) or len(b)!=len(rows) or len({r['video_id'] for r in rows})!=len(rows):
        raise ValueError('Expected one paired question per video')
    rng=np.random.default_rng(2027);delta=a-b;means=[]
    for _ in range(50):means.extend(delta[rng.integers(0,len(rows),size=(200,len(rows)))].mean(1).tolist())
    masks=dict(new_correct=(a==1)&(b==0),degraded=(a==0)&(b==1),both_correct=(a==1)&(b==1),both_wrong=(a==0)&(b==0))
    return dict(questions=len(rows),left_correct=int(a.sum()),right_correct=int(b.sum()),
                net_correct=int(delta.sum()),difference=float(delta.mean()),interval_95=np.quantile(means,[.025,.975]).tolist(),
                **{k:[rows[i]['question_id'] for i in np.flatnonzero(v)] for k,v in masks.items()})


def bank_statistics(labels, keys):
    """接口：真实同视频候选的输入去重与反馈密度；灰帧/异视频不能进入该函数的银行。"""
    if len(labels)!=len(keys):raise ValueError('Bank inputs/labels differ')
    usable=0;pairs=0;conflicts=0;any_correct=0;states=Counter();duplicates=0
    for y,identity in zip(labels,keys):
        if len(y)!=len(identity):raise ValueError('Bank row shape differs')
        reps={}
        for value,key in zip(y,identity):
            if key in reps and reps[key]!=value:raise ValueError('Same complete input has conflicting correctness')
            reps[key]=int(value)
        values=list(reps.values());positive=sum(values);duplicates+=len(y)-len(values)
        state='all_correct' if all(values) else 'all_wrong' if not any(values) else 'mixed'
        states[state]+=1;usable+=state=='mixed';pairs+=positive*(len(values)-positive);any_correct+=bool(positive)
    fixed=np.asarray(labels,dtype=int).sum(0).tolist() if labels else []
    return dict(questions=len(labels),mixed_questions=usable,mixed_fraction=usable/len(labels) if labels else None,
                unique_preference_pairs=pairs,duplicate_candidate_inputs=duplicates,conflicting_inputs=conflicts,
                any_candidate_correct=any_correct,fixed_candidate_correct=fixed,best_fixed_correct=max(fixed) if fixed else None,
                states=dict(states),bank_gap_to_best_fixed=any_correct-max(fixed) if fixed else None)


def _group_summary(items):
    """接口：所有分组均披露实际配对分母，不将小题型结果解释为普遍能力。"""
    result={}
    for condition in CONDITIONS:
        completed=[r['results'][condition] for r in items if r['results'][condition]['status']=='completed']
        result[condition]=dict(questions=len(completed),correct=sum(r['correct'] for r in completed),
                               parse_failures=sum(r['parse_failed'] for r in completed))
    return dict(panel_questions=len(items),conditions=result)


def choose_panel(items):
    """接口：固定题序各取最多4例；先展示新负例、再纠错、再灰/异视频仍对，视频不重复。"""
    selected=[];used=set()
    predicates=[('new_preference',lambda r:r['historical_state']=='all_correct' and any(not r['results'][c]['correct'] for c in ('L','R'))),
                ('corrected_original',lambda r:not r['results']['O']['correct'] and any(r['results'][c]['correct'] for c in ('L','R'))),
                ('controls_still_correct',lambda r:r['results']['G']['correct'] and r['results']['W']['correct'])]
    slots={}
    for kind,predicate in predicates:
        chosen=[]
        for item in items:
            if item['video_id'] in used or not item['reproduced'] or not predicate(item):continue
            chosen.append(item['question_id']);used.add(item['video_id'])
            selected.append(dict(question_id=item['question_id'],category=kind))
            if len(chosen)==4:break
        slots[kind]=dict(maximum=4,selected=chosen,empty_slots=4-len(chosen))
    return dict(cases=selected,slots=slots,maximum_cases=12,maximum_review_minutes=60,
                rule='冻结题序；类别内不补空槽；每视频只出现一次')


def make_board(root,item,plan):
    """接口：现有原图按行画图板；同源时间、局部次序和供体身份仅用于人工审核。"""
    root=Path(root);width=196;cell_height=143;left=72
    board=Image.new('RGB',(left+8*width,45+4*cell_height),'white');draw=ImageDraw.Draw(board)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    except OSError:font=ImageFont.load_default()
    draw.text((8,8),item['question_id']+' | O/W/L/R (G: constant gray)',font=font,fill='black')
    for row,condition in enumerate(('O','W','L','R')):
        y=45+row*cell_height;draw.text((8,y+45),condition,font=font,fill='black')
        record=plan['entries'][condition]['record'];frames=record['frames']
        for column,f in enumerate(frames):
            with Image.open(f['image']) as original:
                image=original.convert('RGB');image.thumbnail((width-4,cell_height-26))
                board.paste(image,(left+column*width+(width-image.width)//2,y))
            t=record['selection']['selected_frames'][column]['timestamp_seconds']
            draw.text((left+column*width,y+cell_height-22),f'{column+1}: {t:.3f}s',font=font,fill='black')
    path=root/'review'/('board_'+item['question_id']+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
    board.save(path,quality=92);return str(path.relative_to(root))


def summarize(root,audit):
    """接口：调用完成后离线计算五条件效果、四→六银行及固定人工面板，不训练任何模型。"""
    root=Path(root);cases=read_json(root/'panel.json')['cases'];evaluation=read_json(root/'evaluation.json');items=[]
    # 步骤1：读取全部原始条件结果，保留O失败后的跳过，不能偷偷补齐分母。
    for case in cases:
        row=case['row'];q=row['question_id'];ev=evaluation[q]
        results={c:read_json(root/'results'/c/(q+'.json')) for c in CONDITIONS}
        same=results['O']['parsed_answer']==ev['historical'][METHODS[0]]['parsed_answer'] and results['O']['parsed_answer'] is not None
        items.append(dict(row,ordinal=case['ordinal'],historical_state=ev['historical_state'],historical=ev['historical'],
                          results=results,reproduced=same,donor_video_id=case['donor']['video_id']))
    valid=[r for r in items if r['reproduced']];rows=[dict(question_id=r['question_id'],video_id=r['video_id']) for r in valid]
    comparisons={c+'_minus_O':paired_comparison(rows,[r['results'][c]['correct'] for r in valid],
                                               [r['results']['O']['correct'] for r in valid]) for c in CONDITIONS[1:]}
    agreement={c:dict(questions=len(valid),same_option=sum(r['results'][c]['parsed_answer']==r['results']['O']['parsed_answer'] for r in valid))
               for c in CONDITIONS[1:]}
    # 步骤2：四→六银行只纳入原四组及新L/R，灰图和异视频绝不伪造成训练负例。
    labels4=[];keys4=[];labels6=[];keys6=[];new_negative=[];new_positive=[];new_correct_over_O=[]
    for item in valid:
        q=item['question_id'];plan=read_json(root/'inputs'/(q+'.json'));ev=evaluation[q]
        y=[int(ev['historical'][m]['correct']) for m in METHODS];key=[ev['historical'][m]['input_key'] for m in METHODS]
        extra=[int(item['results'][c]['correct']) for c in ('L','R')]
        labels4.append(y);keys4.append(key);labels6.append(y+extra);keys6.append(key+[plan['entries'][c]['input_key'] for c in ('L','R')])
        if all(y) and not all(extra):new_negative.append(q)
        if not any(y) and any(extra):new_positive.append(q)
        if not item['results']['O']['correct'] and any(extra):new_correct_over_O.append(q)
    four=bank_statistics(labels4,keys4);six=bank_statistics(labels6,keys6)
    bank=dict(original_candidates=list(METHODS),expanded_candidates=[*METHODS,'L','R'],four=four,six=six,
              new_negative_questions=new_negative,new_positive_opportunities=new_positive,
              new_correct_relative_to_O=new_correct_over_O,
              oracle_count_increase=six['any_candidate_correct']-four['any_candidate_correct'],
              limitation='新增负例可增加排序比较，但不能据此宣称可选输入的答题上界或监督价值提高。')
    # 步骤3：预设统计判断只决定本轮建议；不会触发教师扩容或小头训练。
    reproduction=len(valid)/120
    visual=bool(valid) and all(comparisons[c+'_minus_O']['interval_95'][1]<0 for c in ('G','W'))
    suggestion='reproducibility_insufficient' if reproduction<.95 else 'review_candidate_construction' if visual and new_negative+new_positive else \
               'review_supervision_scenario' if visual else 'original_visual_contribution_uncertain'
    report=dict(status='completed',questions=120,paired_questions=len(valid),original_reproduction_rate=reproduction,
                counts=_group_summary(items),paired_counts=_group_summary(valid),comparisons=comparisons,option_agreement=agreement,
                by_task={t:_group_summary([r for r in items if r['task_type']==t]) for t in sorted({r['task_type'] for r in items})},
                by_historical_state={s:_group_summary([r for r in items if r['historical_state']==s]) for s in ('all_correct','all_wrong','mixed')},
                bank=bank,clear_overall_visual_contribution=visual,preliminary_suggestion=suggestion,audit=audit,
                limitation='固定训练集内部的监督诊断；正确性变化不自动等于语义证据变化，建议须结合有界画面审核。')
    report['paired_by_task']={t:_group_summary([r for r in valid if r['task_type']==t]) for t in sorted({r['task_type'] for r in items})}
    report['paired_by_historical_state']={s:_group_summary([r for r in valid if r['historical_state']==s]) for s in ('all_correct','all_wrong','mixed')}
    panel=choose_panel(items);lookup={r['question_id']:r for r in items}
    for entry in panel['cases']:
        q=entry['question_id'];entry['board']=make_board(root,lookup[q],read_json(root/'inputs'/(q+'.json')))
    durable(root/'case_panel.json',panel);durable(root/'report.json',report)
    # 步骤4：可点击页显示全部120题和每类审核空槽，原始返回通过相对路径可追溯。
    esc=lambda x:html.escape(str(x));page='<!doctype html><meta charset="utf-8"><title>监督画面敏感性</title>'
    page+='<style>body{max-width:1250px;margin:24px auto;font:16px/1.6 sans-serif}img{max-width:100%}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}pre{white-space:pre-wrap}</style>'
    page+='<h1>训练监督对画面内容的敏感性</h1><p>120个训练视频；五条件最多600次诊断。O复现后才运行G/W/L/R；G和W不进入训练银行。</p>'
    page+='<pre>'+esc(json.dumps({k:v for k,v in report.items() if k not in ('by_task','by_historical_state','bank','comparisons')},ensure_ascii=False,indent=2))+'</pre>'
    page+='<p><a href="report.json">完整统计与配对题号</a> · <a href="case_panel.json">固定审核清单及空槽</a></p><h2>120题结果</h2><table><tr><th>题号</th><th>类型</th><th>O</th><th>G</th><th>W</th><th>L</th><th>R</th></tr>'
    for item in items:
        q=item['question_id'];page+='<tr><td>'+esc(q)+'</td><td>'+esc(item['task_type'])+'</td>'
        for c in CONDITIONS:
            result=item['results'][c];label='跳过' if result['status']!='completed' else str(result['parsed_answer'])+(' ✓' if result['correct'] else ' ✗')
            page+='<td><a href="results/'+c+'/'+esc(q)+'.json">'+esc(label)+'</a></td>'
        page+='</tr>'
    page+='</table><h2>有界画面审核</h2>'
    for entry in panel['cases']:
        item=lookup[entry['question_id']];q=item['question_id']
        page+='<article><h3>'+esc(q)+' — '+esc(entry['category'])+'</h3><p>'+esc(item['question'])+'</p>'
        page+='<p>'+esc(' | '.join('ABCDE'[i]+'. '+o for i,o in enumerate(item['options'])))+'</p>'
        page+='<img src="'+entry['board']+'"><details><summary>展开答案和条件结果</summary><pre>'
        page+=esc(json.dumps(dict(answer='ABCDE'[evaluation[q]['answer_index']],results={c:item['results'][c].get('parsed_answer') for c in CONDITIONS}),ensure_ascii=False,indent=2))+'</pre></details></article>'
    (root/'index.html').write_text(page)
    return report
