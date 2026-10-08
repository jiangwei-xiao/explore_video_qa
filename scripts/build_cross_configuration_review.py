"""exp07共同失败八题原帧对照页，选择来自冻结统计而非新增问答。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,html,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,utc
from videoqa_audit.region_scope_review import draw_board
from videoqa_audit.cross_configuration_failures import PROFILES

PANEL=('001-3','007-2','020-2','206-2','022-1','038-2','544-1','003-1')


def build(root):
    """接口：六类首个基线互补题＋局部计数/分散计数对照，图板不是全量归因。"""
    root=Path(root);summary=read_json(root/'summary.json');jobs=[];cases=[]
    # 步骤1：冻结原始共同失败面板和基线来源，不能在看图后更换更有利的题。
    for q in PANEL:
        if q not in summary['common_wrong_ids']:raise ValueError('Panel outside common failure set')
        r=read_json(root/'records'/(q+'.json'));entries=[]
        for p in PROFILES:
            entry=r['conditions'][p];entries.append(dict(profile=p,label='exp07',path=entry['source_result'],correct=False,prediction=entry['prediction']))
        reference=next((dict(profile=p,label=m,path=e['source_result'],correct=True,prediction=e['prediction'])
                       for p in PROFILES for m,e in r['baselines'][p].items() if e['correct']),None)
        if reference:entries.append(reference)
        cases.append(dict(question_id=q,question=r['question'],options=r['options'],gold='ABCD'[r['answer_index']],
                          task_type=r['task_type'],stratum=r['stratum'],entries=entries))
        jobs.extend((str(root),q,e['profile'],e['label'],e['path']) for e in entries)
    with ThreadPoolExecutor(max_workers=2) as executor:boards=list(executor.map(draw_board,jobs))
    # 步骤2：原图、源PTS与历史回答并列，原框架预处理可见性另查，不冒充人工真值。
    page=['<!doctype html><meta charset="utf-8"><title>exp07跨配置共同失败复盘</title><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:24px auto}img{max-width:100%}details{margin:12px 0}</style><h1>exp07跨配置共同失败八题</h1><p>四配置都错；有正确基线时附上固定优先顺序的第一组。原RGB缩略图不是模型处理后的可见性结论。</p>']
    for c in cases:
        page+=['<h2>'+c['question_id']+' '+html.escape(c['task_type'])+'</h2><p>'+html.escape(c['question'])+'</p>',
               '<p>'+html.escape(' / '.join(c['options']))+'</p>']
        for e in c['entries']:
            url=e['profile']+'/'+c['question_id']+'/'+e['label']+'.jpg'
            page+=['<details><summary>'+html.escape(e['profile']+' '+e['label'])+'</summary><a href="'+url+'"><img src="'+url+'"></a><p>历史回答 '+e['prediction']+'；标准 '+c['gold']+'</p></details>']
    page+=['</html>'];p=root/'review/index.html';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(page),encoding='utf-8')
    result=dict(question_ids=list(PANEL),cases=cases,boards=boards,page_sha256=sha256(p),semantic_review='pending',new_model_calls=0,utc=utc())
    durable(root/'review/panel.json',result);return result


def main():
    """接口：只生成既有选帧视图，不执行模型、不写回历史结果。"""
    p=argparse.ArgumentParser();p.add_argument('--audit-id',required=True);a=p.parse_args()
    if not a.audit_id.replace('_','').isalnum():p.error('Invalid audit ID')
    result=build(ROOT/'outputs/analysis'/a.audit_id);print('boards',len(result['boards']))


if __name__=='__main__':main()
