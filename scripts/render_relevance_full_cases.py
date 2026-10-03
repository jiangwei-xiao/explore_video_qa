"""exp12全量验收后的有界案例资产；最多12题，不生成新的问答或证据标签。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json
from videoqa_full.state import durable,rows,exclusive,utc
from videoqa_budget32.relevance_full import METHOD,BASE,REFERENCE
from render_rd_failure_panel import render


def main(run,wait):
    """接口：独立审计通过后生成三组真实输入图板；选题只服从事前规则。"""
    with exclusive(run/'case_render.lock'):
        # 步骤1：后台等待已登记的独立审计进程，不因超时重启任何阶段。
        while not (run/'independent_selection_audit.json').exists():
            if not wait:raise RuntimeError('Independent selection audit pending')
            state=read_json(run/'independent_audit_status.json')
            if state['status']=='paused':raise RuntimeError('Independent audit paused')
            pid=state['pid'];proc=Path('/proc')/str(pid)/'cmdline'
            if not proc.exists():raise RuntimeError('Audit process missing without completion')
            args=proc.read_bytes().decode().split('\0')
            assert '--run' in args and Path(args[args.index('--run')+1]).resolve()==run.resolve()
            durable(run/'case_render_status.json',dict(status='waiting_audit',audit_pid=pid,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            time.sleep(15)
        assert read_json(run/'independent_selection_audit.json')['status']=='passed'
        assert read_json(run/'completion_audit.json')['status']=='passed'
        manifest=read_json(run/'case_manifest.json');ids=manifest['unique_ids'];assert len(ids)==len(set(ids))<=12
        spec=read_json(run/'protocol.json');imported=set(spec['imported_ids']);data={r['question_id']:r for r in rows()}
        directory=run/'case_review';jobs=[]
        for q in ids:
            current=run/('imported/results' if q in imported else 'results')/METHOD/(q+'.json')
            parent=REFERENCE/('imported/results' if q in imported else 'results')/'RD-1.2-exp07'/(q+'.json')
            uniform=BASE/'results/BASE-Uniform'/(q+'.json')
            jobs.extend([(directory,q,METHOD,current),(directory,q,'RD-1.2-exp07',parent),(directory,q,'BASE-Uniform',uniform)])
        # 步骤2：复用精确RGB校验和图板生成；不是模型预处理尺寸下的文字可见性检查。
        with ThreadPoolExecutor(max_workers=2) as executor:
            records=list(executor.map(render,jobs))
        page=['<!doctype html><meta charset="utf-8"><title>RD-1.2-exp12 全量案例</title>',
              '<h1>RD-1.2-exp12 全量固定案例</h1><p>按冻结题序及预定比较选取。原RGB缩略图，非实际模型预处理；语义初审尚未完成。</p>']
        for q in ids:
            row=data[q];page += [f'<h2>{html.escape(q)}</h2><p>{html.escape(row["question"])}</p>',
                '<p>'+html.escape(' / '.join(row['options']))+'</p>']
            for record in [r for r in records if r['question_id']==q]:
                method=record['label'];link=f'{q}/{method}.jpg'
                page += [f'<h3>{html.escape(method)}</h3><a href="{link}"><img style="max-width:100%" src="{link}"></a>',
                    '<details><summary>历史回答</summary>'+html.escape(record['raw_output'])+' / '+str(record['correct'])+'</details>']
        # 步骤3：只声明资产完成，不伪造人工或AI的语义审核。
        (directory/'index.html').write_text('\n'.join(page),encoding='utf-8')
        durable(run/'case_render_status.json',dict(status='completed',questions=len(ids),boards=len(records),
            semantic_review='pending',new_model_calls=0,utc=utc()))
        print('Case assets ready',len(ids),len(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    run=Path(args.run)
    try:main(run,args.wait)
    except BaseException as exc:
        durable(run/'case_render_status.json',dict(status='paused',error=repr(exc),utc=utc()))
        raise
