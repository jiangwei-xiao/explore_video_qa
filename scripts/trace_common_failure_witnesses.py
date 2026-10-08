"""两道共同失败的明确可见状态路径：不扩人工目录、不参与选择。"""
from pathlib import Path
import sys,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.framework_contraction import locations
from videoqa_audit.coverage_objective_review import trace_witness_path,component_at,frame_identity


def trace(root):
    """接口：图板已确认的Pony Express标题和计时图→池/前缀/配额路径；新问答0。"""
    root=Path(root);cases=[]
    # 步骤1：见证从历史Qwen32实际源记录取得，不猜测候选编号或平均FPS。
    for q,time,description in [('007-2',27.261,'Pony Express标题与马匹状态'),('022-1',25.275,'计时图示与黄色扇区')]:
        record=read_json(root/'records'/(q+'.json'))
        witness=min(record['conditions']['qwen32']['frames'],key=lambda f:abs(f['timestamp_seconds']-time));key=frame_identity(witness)
        result=dict(question_id=q,description=description,witness=witness,condition='Posthoc raw-frame visual review; not semantic sufficiency',budgets={})
        # 步骤2：K16为两模型共同输入，只重放一条选择路径；指标全部来自冻结缓存。
        for k in (8,16,32):
            pp,fp,tp,sp=locations(q,k);pool=read_json(pp);saved=read_json(tp);t=saved['exp07'] if k==8 else saved
            f=np.load(fp,allow_pickle=False)
            if k!=32:f=f[pool['original_candidate_indices']]
            path=trace_witness_path(pool,t,[key]);detail=path['paths'][0]
            if 'region_id' in detail:
                region=next(r for r in t['regions'] if r['region_id']==detail['region_id'])
                quota=detail['quota'];position=detail['prefix_position']
                before=region['prefix'][:position-1] if detail['selected'] else region['prefix'][:quota]
                if before:detail['at_current_boundary']=component_at(pool,f,region,detail['candidate_index'],before)
                detail['itm']=pool['initial_scores'][detail['candidate_index']] if detail['candidate_index']<pool['initial_count'] else None
            path['sources']={str(p):sha256(p) for p in (pp,fp,tp,sp)};result['budgets'][str(k)]=path
        cases.append(result)
    value=dict(cases=cases,new_model_calls=0,scope='Two listed case witnesses only; not the 75-fact main denominator')
    durable(root/'witness_paths.json',value);return value


def main():
    """接口：零前向诊断保存到既有分析目录，不写回候选或答案。"""
    p=argparse.ArgumentParser();p.add_argument('--audit-id',required=True);a=p.parse_args()
    if not a.audit_id.replace('_','').isalnum():p.error('Invalid audit ID')
    v=trace(ROOT/'outputs/analysis'/a.audit_id);print(v)


if __name__=='__main__':main()
