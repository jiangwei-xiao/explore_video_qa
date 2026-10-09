"""exp29：固定exp07配额，仅以可见文字状态补充局部重复判定。"""
import re,math
import numpy as np
from videoqa_methods.density_selection import frame_key,validate_candidates
from videoqa_methods.weighted_seed_support import weighted_support


def normalize_text_state(regions):
    """接口：OCR原始框/文字/分数→默认0.5通过框的词集合，不使用问题或答案。

    大小写与弯撇号统一，保留字母/数字和和弦升号；集合不编码词序或语义关系。
    0.5沿用原识别默认值，不搜索阈值；漏检/误识别仍可能影响状态。
    """
    tokens=set()
    # 步骤1：低分原输出在账本保留，选择只消费固定默认通过的可见词。
    for r in regions:
        if not math.isfinite(r['score']) or not 0<=r['score']<=1:raise ValueError('Invalid OCR score')
        if r['score']<0.5:continue
        content=r['text'].replace('’',"'").casefold()
        tokens.update(re.findall(r"[^\W_]+(?:[#♯+'-][^\W_]+)*",content))
    return sorted(tokens)


def text_similarity(left,right):
    """接口：二元文字集合Jaccard，均空为1，单空为0；不是答案相似度。"""
    a,b=set(left),set(right);union=a|b
    return len(a&b)/len(union) if union else 1.0


def text_attenuation(candidate,reference):
    """接口：仅有可靠文字的候选获得状态差异；检测缺失不作为新增信息奖励。"""
    return text_similarity(candidate,reference) if candidate else 1.0


def select_text_state(pool,features,parent_trace,states,frame_budget):
    """接口：合法父池/视觉特征/冻结配额/OCR词集→局部新前缀与同预算输入。

    有字候选只改D=max_s[cos_+(i,s)×J(T_i,T_s)]；无字候选仍使用原D。
    原L、锚点、区域及配额全部不变；这是有方向的重复抑制，不称全空间距离。
    OCR文本不加入最终QA提示词；视觉与文字不同不自动代表新增答题信息。
    """
    c=pool['candidates'];validate_candidates(c,pool['video']);f=np.asarray(features,dtype=np.float32)
    if len(states)!=len(c) or f.ndim!=2 or len(f)!=len(c) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):raise ValueError('State or feature identity differs')
    normalized=[set(s) for s in states];records=[];picked=[]
    # 步骤1：消费冻结exp07唯一候选归属，避免同时更改区域、配额或支持。
    for original,region in zip(parent_trace['regions'],pool['plan']['regions']):
        ids=original['members'];anchor=region['anchor_index'];quota=parent_trace['region_budgets'][str(region['region_id'])]
        if original['region_id']!=region['region_id'] or not 1<=quota<=len(ids) or anchor not in ids:raise ValueError('Parent allocation differs')
        L,_,_=weighted_support(f,ids,region['seed_indices'],pool['initial_scores'])
        chosen=[anchor];vis=np.clip(f[ids]@f[anchor],0,1)
        txt=np.array([text_attenuation(normalized[i],normalized[anchor]) for i in ids],dtype=np.float32)
        D=vis*txt;nearest=np.full(len(ids),anchor,dtype=np.int64);steps=[]
        # 步骤2：先乘帧对两种相似，再对已选参照取max，不能交换乘法与max顺序。
        while len(chosen)<min(frame_budget,len(ids)):
            options=[]
            for j,i in enumerate(ids):
                if i in chosen:continue
                options.append(dict(candidate_index=i,support=float(L[j]),redundancy=float(D[j]),
                    redundancy_reference=int(nearest[j]),local_score=float(L[j])*math.sqrt(1-float(D[j]))))
            winner=min(options,key=lambda p:(-p['local_score'],*frame_key(c[p['candidate_index']])))
            i=winner['candidate_index'];chosen.append(i);steps.append(winner)
            visual=np.clip(f[ids]@f[i],0,1)
            textual=np.array([text_attenuation(normalized[j],normalized[i]) for j in ids],dtype=np.float32)
            joint=visual*textual;improved=joint>D;nearest[improved]=i;D=np.maximum(D,joint)
        picked+=chosen[:quota]
        records.append(dict(original,prefix=chosen,local_steps=steps,parent_prefix=original['prefix'],quota=quota,
            coverage_curve=None,coverage_not_used='parent quotas frozen; no budget reoptimization'))
    # 步骤3：预算、锚点和源顺序闭合，候选来源与OCR内容不会直接产生保留名额。
    selected=sorted(picked,key=lambda i:frame_key(c[i]));target=min(frame_budget,len(c))
    if len(selected)!=target or len(set(selected))!=target:raise ValueError('Fixed budget mismatch')
    if sorted(i for r in records for i in r['members'])!=list(range(len(c))):raise ValueError('Candidate ownership differs')
    return selected,dict(regions=records,region_budgets=parent_trace['region_budgets'],
        protected_indices=[r['anchor_index'] for r in pool['plan']['regions']],selected_indices=selected,
        rule='exp07 fixed allocation; visual cosine times OCR-token Jaccard pairwise redundancy',
        limitation='OCR errors and new text can reward irrelevant states; no claim of semantic sufficiency')
