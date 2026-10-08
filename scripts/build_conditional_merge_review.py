"""exp28开发得失五题对照页；答题变化与源画面变化分开呈现。"""
from pathlib import Path
import sys,argparse,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable
from videoqa_audit.framework_contraction import RUN32,RUN8
from videoqa_audit.region_scope_review import draw_board

CASES=(('103-1','qwen32'),('844-1','qwen32'),('556-1','qwen32'),('260-2','llava8'),('204-3','llava8'))


def build(root):
    """接口：固定改善/退化与已知见证题，源图板和全部真实回答可点击。"""
    root=Path(root);bank={r['question_id']:r for r in rows()};items=[]
    page=['<!doctype html><meta charset="utf-8"><title>exp28条件合并结果复核</title><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:24px auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%}</style><h1>exp28条件合并结果复核</h1><p>Qwen32 130/200，LLaVA8 119/200，触发停止条件，未运行全量。原RGB缩略图；答错不直接证明关键证据被移除。</p>']
    # 步骤1：两个不同职责的区间记录仍保留，不能把源帧内部region_id当新选择组。
    for q,p in CASES:
        base=RUN32 if p=='qwen32' else RUN8;before=base/'selections/RD-1.2-exp07'/(q+'.json');after=root/'offline'/p/(q+'.json')
        new=read_json(after);answer=read_json(root/'qa'/p/'results/RD-1.2-exp28'/(q+'.json'))
        for label,path in [('exp07',before),('exp28',after)]:draw_board((str(root),q,p,label,str(path)))
        row=bank[q];page+=['<h2>'+q+' '+p+'</h2><p>'+html.escape(row['question'])+'</p><p>'+html.escape(' / '.join(row['options']))+'</p>',
            '<p>选择组 '+str(new['original_region_count'])+' → '+str(new['selection_group_count'])+'；新回答 '+answer['answer']['parsed_answer']+'，标准 '+'ABCD'[row['answer_index']]+'</p><div class="pair">']
        for label in ('exp07','exp28'):
            url=p+'/'+q+'/'+label+'.jpg';page+=['<div>'+label+'<a href="'+url+'"><img src="'+url+'"></a></div>']
        page+=['</div>'];items.append(dict(question_id=q,profile=p,selection_sha256=sha256(after),result_sha256=sha256(root/'qa'/p/'results/RD-1.2-exp28'/(q+'.json'))))
    page+=['<p>完整配对得失见运行目录summary.json；同帧导入不是重复性验证。</p>'];out=root/'review/index.html';out.write_text('\n'.join(page),encoding='utf-8')
    durable(root/'review/cases.json',dict(cases=items,page_sha256=sha256(out),new_model_calls=0))


def main():
    """接口：只读已完成200题结果，不添加问答或修改实验输入。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    build(ROOT/'outputs/methods'/a.run_id)


if __name__=='__main__':main()
