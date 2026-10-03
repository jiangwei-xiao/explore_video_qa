"""将已固定窗外支持规则原样扩到开发200题，只做离线验证，不调用问答。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import probe_external_support as probe
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive
RUN=ROOT/'outputs/analysis/external_support200_20261003_r1'
PILOT=ROOT/'outputs/analysis/external_support12_20261003_r1/selections.json'
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def main():
    """接口：只替换独立输入清单，原提议及选择函数不改；标签只在选择后计代理指标。"""
    with exclusive(RUN/'expansion.lock'):
        pilot=read_json(PILOT)
        for p,h in pilot['sources'].items():assert sha256(p)==h,p
        ids=[r['question_id'] for r in read_json(DATA)['rows']];assert len(ids)==len(set(ids))==200
        manifest=dict(cases=[dict(question_id=q) for q in ids],dataset_manifest=str(DATA),dataset_sha256=sha256(DATA),
            expansion_runner_sha256=sha256(__file__),pilot_sha256=sha256(PILOT),new_model_calls=0)
        pp=RUN/'input_manifest.json'
        if pp.exists():assert read_json(pp)==manifest
        else:durable(pp,manifest)
        # 步骤1：独立进程中仅换清单变量，不编辑固定12题代码、协议或原始结果。
        probe.PANEL=pp
        probe.main(RUN)
        result=read_json(RUN/'selections.json');records={r['question_id']:r for r in result['records']}
        overlap=0
        for r in pilot['records']:
            if r['question_id'] in records:
                x=records[r['question_id']]
                assert x==r,r['question_id'];overlap+=1
        # 步骤2：冻结75事实只用于评价，新增/丢失精确见证不直接等同语义得失。
        facts=read_json(FACTS)['entries'];assert len(facts)==75 and len({r['question_id'] for r in facts})==42
        evaluated=[]
        for fact in facts:
            q=fact['question_id'];r=records[q];pool=read_json(probe.BASE/'pools'/(q+'.json'));c=pool['candidates'];w=set(fact['witness_pts'])
            evaluated.append(dict(question_id=q,fact_id=fact['fact_id'],
                old=bool(w&{c[i]['source_pts'] for i in r['old_indices']}),
                new=bool(w&{c[i]['source_pts'] for i in r['selected_indices']})))
        summary=dict(questions=200,pilot_overlap_exact=overlap,changed_questions=result['changed_questions'],
            changed_positions=sum(len(r['added']) for r in records.values()),
            old_exact_hits=sum(r['old'] for r in evaluated),new_exact_hits=sum(r['new'] for r in evaluated),
            gained=[r for r in evaluated if r['new'] and not r['old']],lost=[r for r in evaluated if r['old'] and not r['new']],
            facts=evaluated,fact_protocol_sha256=sha256(FACTS),new_model_calls=0,new_answers=0,
            scope='Frozen development200; exact witness proxy only; no automatic QA gate')
        out=RUN/'summary.json'
        if out.exists():assert read_json(out)==summary
        else:durable(out,summary)
        print({k:v for k,v in summary.items() if k!='facts'},flush=True)


if __name__=='__main__':main()
