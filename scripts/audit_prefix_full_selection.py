"""全量exp07选择的独立闭合检查；只读候选与轨迹，不改变输入、不调用模型。"""
import argparse
from pathlib import Path
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_budget32.prefix_full import BASE,DEV,METHOD


def optimal_value(regions, budget):
    """接口：独立二维数值表重算最优值，不调用生产分配函数或使用其配额。"""
    table = [[-math.inf]*(budget+1) for _ in range(len(regions)+1)]
    table[0][0] = 0.0
    # 步骤1：逐区域、总名额和可行前缀长度枚举，不假设覆盖曲线为凹函数。
    for r,region in enumerate(regions,1):
        curve,w = region['coverage_curve'],region['density_weight']
        assert all(math.isfinite(x) and 0<=x<=1 for x in curve)
        assert all(b>=a for a,b in zip(curve,curve[1:])) and w>0
        for used in range(1,budget+1):
            table[r][used] = max((table[r-1][used-n]+w*curve[n] for n in range(1,min(used,len(curve)-1)+1)),default=-math.inf)
    return table[-1][-1]


def main(run):
    """接口：完整2700题选择身份、粗参照闭合、实际前缀与200题复用身份核验。"""
    run = Path(run); summary = read_json(run/'selection_summary.json'); assert summary['status']=='passed'
    counts = {}; imported = set(summary['imported_ids']); checked = 0
    for row in rows():
        q = row['question_id']; rel = f'selections/{METHOD}/{q}.json'
        assert sha256(run/rel)==summary['exports'][rel]
        item = read_json(run/rel); trace = read_json(run/item['trace_file'])
        assert sha256(run/item['trace_file'])==item['trace_sha256']
        pool = read_json(BASE/'pools'/(q+'.json')); target = min(32,len(pool['candidates']))
        # 步骤2：粗参照与全部候选分别恰好覆盖一次，预算必须来自实际局部前缀。
        members = [i for r in trace['regions'] for i in r['members']]
        refs = [i for r in trace['regions'] for i in r['reference_coarse_indices']]
        assert sorted(members)==list(range(len(pool['candidates'])))
        assert sorted(refs)==list(range(pool['initial_count']))
        chosen = []
        for r in trace['regions']:
            n = trace['region_budgets'][str(r['region_id'])]
            assert 1<=n<=len(r['prefix']) and len(set(r['prefix']))==len(r['prefix'])
            assert set(r['prefix'])<=set(r['members'])
            chosen.extend(r['prefix'][:n])
        assert len(chosen)==len(set(chosen))==target
        expected = sorted(chosen,key=lambda i:(pool['candidates'][i]['source_pts'],pool['candidates'][i]['candidate_index']))
        assert expected==item['selected_indices']
        assert [pool['candidates'][i] for i in expected]==item['selection']['selected_frames']
        assert abs(optimal_value(trace['regions'],target)-trace['objective'])<1e-10
        assert set(trace['protected_indices'])<=set(chosen)
        if q in imported:
            old = read_json(DEV/'selections'/METHOD/(q+'.json'))
            assert item['selection']==old['selection'] and item['rgb_sha256']==old['rgb_sha256']
        checked+=1; counts[target]=counts.get(target,0)+1
    assert checked==2700 and len(imported)==200
    durable(run/'independent_selection_audit.json',dict(status='passed',questions=checked,development_replayed=200,
        frame_counts=counts,new_model_calls=0,checks=['source identity','unique membership','coarse reference scope','prefix budget closure','independent DP objective','development RGB identity'],utc=utc()))
    print('Independent selection audit passed:',checked,counts,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run',required=True)
    main(parser.parse_args().run)
