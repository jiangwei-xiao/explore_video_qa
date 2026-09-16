"""登记实际查看过的384输入图板；不自动把生成图板当成完成人工审核。"""
import argparse
import sys
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import read_json,write_json,sha256


def record(audit,spec):
    """输入人工明确列出的页码/观察，保存图板哈希、PTS覆盖与尚不清楚的信息。"""
    audit=Path(audit);root=audit/'full_review'/'visibility';root.mkdir(exist_ok=True)
    questions={q['question_id']:q for q in read_json(audit/'questions.json')}
    # 步骤1：仅登记本次人工明确确认的页面，不把未查看页面补为已读。
    for qid,item in spec.items():
        q=questions[qid];target=root/f'{qid}.json';previous=read_json(target) if target.exists() else dict(question_id=qid,pages={})
        # 步骤1a：先保留旧版，后续修订不抹掉此前的观察及争议理由。
        if target.exists():
            history=root/'history'/qid;history.mkdir(parents=True,exist_ok=True)
            write_json(history/f'{uuid4().hex}.json',previous)
        for page in item['pages']:
            start=(page-1)*16;rows=q['frames'][start:start+16]
            if page<1 or not rows:raise ValueError('页码越界')
            path=audit/'full_review'/'boards'/qid/f'{page:02}.png'
            previous['pages'][str(page)]=dict(board_file=str(path.relative_to(audit)),sha256=sha256(path),
                source_pts=[r['source_pts'] for r in rows],reviewer='assistant',observation=item['observation'],
                uncertainties=item.get('uncertainties',[]),
                reviewed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(target,previous)
    # 步骤2：按唯一PTS计覆盖，审核过但不清楚的帧仍记录不确定，不假装可读。
    count=0;complete=[]
    for qid,q in questions.items():
        path=root/f'{qid}.json';seen=set()
        if path.exists():
            for entry in read_json(path)['pages'].values():
                if sha256(audit/entry['board_file'])!=entry['sha256']:raise ValueError('已审图板已改变')
                seen.update(entry['source_pts'])
        count+=len(seen)
        if seen=={r['source_pts'] for r in q['frames']}:complete.append(qid)
    write_json(root/'progress.json',dict(reviewed_unique_frames=count,total_unique_frames=2235,
               complete_questions=complete,complete_question_count=len(complete),total_questions=50,
               note='384图板已查看不等于每处文字清晰，更不证明模型编码后能够正确理解。'))
    print(count,'/2235;',len(complete),'/50')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--spec',required=True)
    a=p.parse_args();record(a.audit,read_json(a.spec))
