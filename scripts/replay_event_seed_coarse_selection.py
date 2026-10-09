"""事件入口的粗池选帧路径诊断：统一无细帧控制，不直接改变正式exp07。"""
from pathlib import Path
from fractions import Fraction
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from probe_event_clause_localization import TARGETS
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_methods.budget_contract import budget_config,select_exp07
from videoqa_methods.density_selection import build_regions,source_time
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.coarse_only_ablation import feasible_reference
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage

SOURCE=ROOT/'outputs/analysis/event_clause_full_coarse__videomme5__20261009_r1'


def locator_only(coarse,features,event_scores):
    """接口：只以事件分数建立种子/区域，原问题支持、视觉差异与等权DP不变。

    无细帧仅用于明确入口是否传递到最终选择；不能把本诊断当作完整方法。
    原题支持仍可能压低属性载体，新事件入口不自动取得最终保留资格。
    """
    cfg=budget_config(32);plan=build_regions(coarse['candidates'],event_scores,coarse['video'],cfg)
    cells=coarse_selection_cells(plan,coarse['video'])['regions']
    # 步骤1：构造固定选择器的可行参照，不增新分配规则或目标权重。
    caps=[sum(Fraction(r['left_fraction'])<=source_time(c,coarse['video'])<=Fraction(r['right_fraction']) for c in coarse['candidates']) for r in cells]
    counts=feasible_reference(caps,min(32,len(coarse['candidates'])))
    reference={str(r['region_id']):n for r,n in zip(cells,counts)}
    # 步骤2：原题得分只消费为原支持函数；事件得分不成为最终留帧优先级。
    selected,trace=select_prefix_coverage(coarse['candidates'],features,plan,coarse['video'],cfg,reference,coarse['initial_scores'])
    return selected,trace,plan


def replay():
    """接口：五题原/事件入口同粗池比较，保存源帧、路径及配额；新增模型调用0。"""
    from videoqa_audit.framework_contraction import BASE
    root=SOURCE/'coarse_selection_replay';root.mkdir(parents=True,exist_ok=True);records=[];files={}
    for q,witness in TARGETS.items():
        inp=read_json(SOURCE/'inputs'/(q+'.json'));pool=read_json(inp['pool_path']);result=read_json(SOURCE/'results'/(q+'.json'))
        n=pool['initial_count'];feature_path=BASE/'features'/(q+'.npy');features=np.load(feature_path,allow_pickle=False)[:n]
        coarse=dict(pool,candidates=pool['candidates'][:n])
        # 步骤1：同一完整粗池及纯视觉特征，控制与候选均不借用旧补查帧。
        original,control=select_exp07(coarse,features,32)
        changed,trace,plan=locator_only(coarse,features,result['comparisons']['event_clause']['scores'])
        details={}
        for label,selected,t in [('original_locator_coarse_only',original,control),('event_locator_coarse_only',changed,trace)]:
            region=next(r for r in t['regions'] if witness in r['members']);position=region['prefix'].index(witness)+1 if witness in region['prefix'] else None
            quota=t['region_budgets'][str(region['region_id'])]
            value=dict(question_id=q,condition=label,selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in selected]),
                witness_index=witness,witness_selected=witness in selected,local_prefix_position=position,quota=quota,trace=t,
                new_model_calls=0,coarse_only=True,source_initial_count=n,
                sources={str(Path(inp['pool_path'])):sha256(inp['pool_path']),str(feature_path):sha256(feature_path),str(SOURCE/'results'/(q+'.json')):sha256(SOURCE/'results'/(q+'.json'))})
            path=root/label/(q+'.json')
            if path.exists() and read_json(path)!=value:raise ValueError('Coarse path replay changed')
            if not path.exists():durable(path,value)
            files[str(path)]=sha256(path);details[label]={k:value[k] for k in ('witness_selected','local_prefix_position','quota')}
        records.append(dict(question_id=q,paths=details,changed_frames=len(set(original)^set(changed))//2,
            scope='No-refinement diagnostic, not comparison with full historic QA input'))
    # 步骤2：候选资格/定位/局部次序/名额区分，任何答题能力结论均留空。
    summary=dict(records=records,files=files,new_model_calls=0,code_sha256=sha256(Path(__file__)),scope='Two coarse-only paths, no QA accuracy')
    durable(root/'summary.json',summary);return summary


if __name__=='__main__':replay()
