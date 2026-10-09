"""exp31：弱文字新奇信号只竞争一个原局部末位，不重排主体。"""
import math
import numpy as np
from videoqa_methods.density_selection import frame_key
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.text_state_redundancy import text_attenuation


def select_bounded_text_state(pool,features,parent_trace,states,frame_budget):
    """接口：父输入＋原OCR词→至多一次替换；锚点、配额和其他全部源帧固定。

    每个有额外配额的区域提出一个文字候选，只可替代原前缀该区最后一帧；
    全题仅选最大正增量的一处。1是最小干预步长，不是经题目优化的预算比例。
    它不保证末位是语义最弱帧，也不保证OCR新词提供答案，必须记录所有取舍。
    """
    c=pool['candidates'];f=np.asarray(features,dtype=np.float32)
    if len(states)!=len(c) or len(f)!=len(c):raise ValueError('Text/source correspondence differs')
    original={i for r in parent_trace['regions'] for i in r['prefix'][:parent_trace['region_budgets'][str(r['region_id'])]]}
    if len(original)!=min(frame_budget,len(c)):raise ValueError('Parent budget differs')
    proposals=[]
    # 步骤1：固定原绝大部分已选参照，不能借文字优势连带移除其他已有状态。
    for r,region in zip(parent_trace['regions'],pool['plan']['regions']):
        quota=parent_trace['region_budgets'][str(r['region_id'])]
        if quota<=1:continue
        removed=r['prefix'][quota-1];fixed=r['prefix'][:quota-1];ids=r['members']
        support,_,_=weighted_support(f,ids,region['seed_indices'],pool['initial_scores']);pos={i:j for j,i in enumerate(ids)}
        def utility(i):
            """仅评价这一个空位；帧对视觉×文字后取max，空文字保持视觉判断。"""
            D=max(float(np.clip(f[i]@f[s],0,1))*text_attenuation(states[i],states[s]) for s in fixed)
            return float(support[pos[i]])*math.sqrt(1-D),D
        incumbent,old_D=utility(removed)
        options=[i for i in ids if i not in original and states[i]]
        if not options:continue
        winner=min(options,key=lambda i:(-utility(i)[0],*frame_key(c[i])))
        score,D=utility(winner)
        proposals.append(dict(region_id=r['region_id'],removed_index=removed,added_index=winner,
            gain=score-incumbent,new_utility=score,old_utility=incumbent,new_redundancy=D,old_redundancy=old_D,
            fixed_indices=fixed,old_tokens=states[removed],new_tokens=states[winner]))
    # 步骤2：一题至多一处正增量替换，未改善则精确保持父选择。
    viable=[p for p in proposals if p['gain']>0]
    winner=min(viable,key=lambda p:(-p['gain'],*frame_key(c[p['added_index']]))) if viable else None
    selected=set(original)
    if winner:selected.remove(winner['removed_index']);selected.add(winner['added_index'])
    protected={r['anchor_index'] for r in pool['plan']['regions']}
    if len(selected)!=len(original) or not protected<=selected:raise ValueError('Bounded protection or budget differs')
    # 步骤3：恢复时间顺序，来源和文本不加入问答提示，候选与预算均不扩大。
    result=sorted(selected,key=lambda i:frame_key(c[i]))
    return result,dict(proposals=proposals,winner=winner,parent_selected_indices=sorted(original),selected_indices=result,
        protected_indices=sorted(protected),region_budgets=parent_trace['region_budgets'],maximum_replacements=1,
        rule='One largest positive local text-state utility change; original prefix last slot only',
        limitation='Weak cue bounded, not calibrated QA value; last slot may still contain unique evidence')
