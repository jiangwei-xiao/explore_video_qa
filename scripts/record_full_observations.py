"""展开全量人工观察；标签只能来自明确填写的已查看帧，禁止按分数或答题对错填充。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.core import read,write
from videoqa_audit.observations import apply_observations,indices


def record(out,specifications):
    """将显式人工组级判据应用于已选PTS组合，保存判据及展开结果。"""
    out=Path(out);questions={q['question_id']:q for q in read(out/'questions.json')}
    for qid,s in specifications.items():
        q=questions[qid];lookup={r['source_pts']:i for i,r in enumerate(q['frames'],1)}
        groups={}
        # 步骤1：先按人工确认的缺失项给出默认组级判断；特殊充分性判据也由人工指定。
        for alias,pts in q['groups'].items():
            selected={lookup[p] for p in pts};assessment=s['group_default']
            for rule in s.get('group_rules',[]):
                if all(selected.intersection(indices(condition)) for condition in rule['requires_any_from_each']):
                    assessment=rule['assessment']
            groups[alias]=assessment
        expanded=dict(observations=s['observations'],same_evidence_units=s.get('same_evidence_units',[]),
                      groups=groups,notes=s['notes'],source_checks=s.get('source_checks',[]),input_visibility=s.get('input_visibility',{}))
        # 步骤2：保留人工条件与任务/证据双轴；展开接口会拒绝任何漏标帧。
        write(out/'full_review'/'observations'/f'{qid}.json',s)
        apply_observations(out,{qid:expanded},revision_reason='全量审计阶段：按已查看原帧/输入视图和记录的判据修订')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--spec',required=True)
    a=p.parse_args();record(a.out,read(a.spec))
