"""exp28独立完成审计：图连通合并、唯一归属、预算、调用与恢复。"""
from pathlib import Path
from fractions import Fraction
import sys,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable
from videoqa_audit.framework_contraction import locations


def expected_groups(cores):
    """接口：独立两两相交图求连通分量，稀疏核心没有合并边。"""
    parent={c['core_id']:c['core_id'] for c in cores}
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    # 步骤1：两端均触发且闭窗口相交才有边，不使用生产版排序合并循环。
    for a in cores:
        for b in cores:
            if a['core_id']>=b['core_id'] or min(a['seed_count'],b['seed_count'])<3:continue
            if max(Fraction(a['left_fraction']),Fraction(b['left_fraction']))<=min(Fraction(a['right_fraction']),Fraction(b['right_fraction'])):
                x,y=find(a['core_id']),find(b['core_id']);parent[max(x,y)]=min(x,y)
    groups={}
    for c in cores:groups.setdefault(find(c['core_id']),[]).append(c['core_id'])
    return sorted(sorted(g) for g in groups.values())


def check(root):
    """接口：5400条实际离线轨迹和400份实际回答闭合；不生成、不修改旧产物。"""
    root=Path(root);sources={};checked=0
    # 步骤1：源池的原核心决定预期组；同一候选仅属于最近核心对应组。
    for profile,k in [('qwen32',32),('llava8',8)]:
        for row in rows():
            q=row['question_id'];record_path=root/'offline'/profile/(q+'.json');item=read_json(record_path)
            pp,fp,tp,sp=locations(q,k);pool=read_json(pp);trace_path=root/item['trace_file'];trace=read_json(trace_path)
            for p,h in item['sources'].items():
                if sha256(p)!=h:raise ValueError('Frozen source changed')
            if sha256(trace_path)!=item['trace_sha256']:raise ValueError('Saved trace changed')
            groups=expected_groups(pool['plan']['cores'])
            if sorted(sorted(r['core_ids']) for r in trace['regions'])!=groups:raise ValueError('Conditional graph differs')
            mapping=item['parent_candidate_indices'];candidates=[pool['candidates'][i] for i in mapping]
            all_members=[i for r in trace['regions'] for i in r['members']]
            if sorted(all_members)!=list(range(len(candidates))):raise ValueError('Candidate ownership lost or duplicated')
            cores=pool['plan']['cores'];intervals={c['core_id']:(min(pool['candidates'][i]['source_pts'] for i in c['seed_indices']),max(pool['candidates'][i]['source_pts'] for i in c['seed_indices'])) for c in cores}
            for index,frame in enumerate(candidates):
                owner=min(intervals,key=lambda j:(max(intervals[j][0]-frame['source_pts'],0,frame['source_pts']-intervals[j][1]),intervals[j][0],j))
                if trace['candidate_core_owners'][index]!=owner:raise ValueError('PTS owner differs')
            ids=[]
            # 步骤2：独立前缀预算DP核对；每区锚点、粗参考和密度按实际新组派生。
            states={0:(0.,())}
            for r in trace['regions']:
                if r['prefix'][0]!=r['anchor_index'] or r['density_weight']!=1+sum(cores[j]['seed_count']-1 for j in r['core_ids']):raise ValueError('Anchor or density differs')
                if r['reference_coarse_indices']!=[i for i in r['members'] if i<pool['initial_count']]:raise ValueError('Coarse reference differs')
                new={}
                for used,(score,allocation) in states.items():
                    for n in range(1,min(len(r['prefix']),item['actual_frame_count']-used)+1):
                        proposal=(score+r['density_weight']*r['coverage_curve'][n],allocation+(n,))
                        if proposal>new.get(used+n,(-float('inf'),())):new[used+n]=proposal
                states=new
                quota=trace['region_budgets'][str(r['region_id'])]
                if not 1<=quota<=len(r['prefix']):raise ValueError('Quota outside prefix')
                ids+=r['prefix'][:quota]
            value,allocation=states[item['actual_frame_count']]
            if allocation!=tuple(trace['region_budgets'][str(r['region_id'])] for r in trace['regions']) or abs(value-trace['objective'])>1e-9:raise ValueError('DP solution differs')
            keys=lambda frames:sorted((f['source_pts'],f['source_frame_index']) for f in frames)
            if len(ids)!=len(set(ids)) or keys([candidates[i] for i in ids])!=keys(item['selection']['selected_frames']):raise ValueError('Evaluated input differs from prefix selection')
            sources[str(record_path)]=sha256(record_path);sources[str(trace_path)]=sha256(trace_path);checked+=1
    # 步骤3：主执行器的实际模型观测/账本独立复核，计分停止条件不得改写。
    from videoqa_audit.conditional_merge_runtime import audit,report
    before={p:{str(path):sha256(path) for folder in ('results','calls','raw_generations') for path in (root/'qa'/p/folder).rglob('*.json')} for p in ('qwen32','llava8')}
    for p in before:audit(root/'qa'/p)
    result=report(root)
    after={p:{str(path):sha256(path) for folder in ('results','calls','raw_generations') for path in (root/'qa'/p/folder).rglob('*.json')} for p in before}
    if before!=after:raise ValueError('Recovery changed actual generation artifacts')
    summary=dict(status='passed',offline_records=checked,method_results=400,new_calls=sum(p['new_calls'] for p in result['profiles'].values()),
                 imports=sum(p['imported'] for p in result['profiles'].values()),generation_artifacts=sum(len(x) for x in before.values()),
                 extra_generations=0,sources=sources,formal_artifacts=before,gate_stopped=result['development_stop_for_clear_drop'])
    durable(root/'independent_completion_audit.json',summary);return summary


def main():
    """接口：完成态验证已评测输入，不加载GPU或扩大调用额度。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    result=check(ROOT/'outputs/methods'/a.run_id);print({k:v for k,v in result.items() if k not in ('sources','formal_artifacts')})


if __name__=='__main__':main()
