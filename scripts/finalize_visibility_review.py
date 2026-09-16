"""逐题核实384查看日志与实际图板，汇总可点击复核说明；不宣布整个证据审计完成。"""
import argparse
import html
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from videoqa_runtime.common import read_json, write_json, sha256


def finalize(audit):
    """输入固定审计目录，验证每一页的PTS/哈希并输出本阶段清单及HTML、Markdown。"""
    audit=Path(audit); questions=read_json(audit/'questions.json'); count=0; boards=[]; entries=[]
    # 步骤1：逐题按真实图板顺序验证覆盖，不能仅信progress中的总数。
    for q in questions:
        qid=q['question_id']; record=read_json(audit/'full_review'/'visibility'/f'{qid}.json')
        expected_pages={str(i) for i in range(1, math.ceil(len(q['frames'])/16)+1)}
        if set(record['pages'])!=expected_pages:raise ValueError(f'{qid}页面缺失/额外页面')
        seen=set(); observations=[]; doubts=[]; links=[]
        for key in sorted(expected_pages,key=int):
            page=record['pages'][key]; start=(int(key)-1)*16
            expected=[f['source_pts'] for f in q['frames'][start:start+16]]
            if page['source_pts']!=expected or seen.intersection(expected):raise ValueError(f'{qid}页内PTS不匹配或重复')
            file=audit/page['board_file']
            if sha256(file)!=page['sha256']:raise ValueError(f'{qid}图板哈希变化')
            if not page['observation'].strip():raise ValueError(f'{qid}缺少观察记录')
            seen.update(expected); boards.append({'question_id':qid,'page':int(key),'file':page['board_file'],'sha256':page['sha256']})
            if page['observation'] not in observations:observations.append(page['observation'])
            for doubt in page.get('uncertainties',[]):
                if doubt not in doubts:doubts.append(doubt)
            links.append(f'../boards/{qid}/{int(key):02}.png')
        if seen!={f['source_pts'] for f in q['frames']}:raise ValueError(f'{qid}联合帧未闭合')
        count+=len(seen); entries.append({'question_id':qid,'question':q['question'],'frame_count':len(seen),'observations':observations,'uncertainties':doubts,'board_links':links})
    if len(questions)!=50 or count!=2235:raise ValueError('冻结50题2235唯一帧覆盖不符')
    # 步骤2：保存独立版本，原标签、旧验证与历史问答不改写。
    out=audit/'full_review'/'visibility_r4'; out.mkdir(exist_ok=True)
    summary={'status':'all_384_boards_reviewed_not_full_audit_complete','questions':len(questions),'unique_frames':count,'boards':len(boards),'questions_sha256':sha256(audit/'questions.json'),'model_calls':0,'reviewer':'assistant','review_mode':'posthoc_initial_review','human_agreement':None,'remaining':'全量事实等价/机制归因及最终方案收敛'}
    write_json(out/'verification.json',summary); write_json(out/'board_manifest.json',boards); write_json(out/'observations.json',entries)
    # 步骤3：输出只含题面与观察的可点击页面，不显示方法预测或伪造用户裁决。
    lines=['# 全50题384输入可见性复核','','50题、2235张唯一帧、逐页图板已核对。事后AI初审，不是用户金标准；不确定不等于未查看。','']
    blocks=['<!doctype html><meta charset="utf-8"><title>384输入复核</title><style>body{max-width:1000px;margin:32px auto;font:16px/1.7 sans-serif;padding:0 20px}section{border-top:1px solid #ddd;padding:20px 0}a{margin-right:15px}</style><h1>全50题384输入可见性复核</h1><p>事后AI初审；2235张唯一输入帧全部查看。可见不等于模型理解，未确认事项保留；整个机制审计尚未完成。</p>']
    for e in entries:
        qid=e['question_id']; lines += [f'## {qid}', '', e['question'], '', *e['observations'], '']
        lines += ['不确定：'+d for d in e['uncertainties']]+['']
        lines += [' / '.join(f'[第{i+1}页]({u})' for i,u in enumerate(e['board_links'])), '']
        blocks += [f'<section id="{qid}"><h2>{qid}</h2><p>{html.escape(e["question"])}</p>']
        blocks += [f'<p>{html.escape(t)}</p>' for t in e['observations']]
        blocks += [f'<p>不确定：{html.escape(t)}</p>' for t in e['uncertainties']]
        blocks += [f'<a href="{u}">第{i+1}页</a>' for i,u in enumerate(e['board_links'])]+['</section>']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8'); (out/'index.html').write_text('\n'.join(blocks),encoding='utf-8')
    print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);a=p.parse_args();finalize(a.audit)
