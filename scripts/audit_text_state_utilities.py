"""只读复算exp31实际竞争分量：字符串变化如何放松视觉近重复惩罚。"""
import argparse
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive
from videoqa_audit.framework_contraction import locations
from videoqa_audit.text_state_experiments import PILOT
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.text_state_redundancy import text_attenuation
from videoqa_methods.density_selection import frame_key

SOURCE = ROOT/'outputs/methods/rd12_exp31_bounded_text_state_20261009_r1'
CONFIRMED = ROOT/'outputs/methods/rd12_exp30_confirmed_text_state_20261009_r1'


def companion_similarity(features, states, index, reference):
    """接口：仅作零调用软描述对照，视觉和单位词向量串接后的余弦。

    等范数描述是待检验的建模假设，不是信息价值定理；它避免文字差异把
    视觉相似直接清零，但不能识别问题是否需要这些词，不接入正式选择器。
    """
    a, b = set(states[index]), set(states[reference])
    visual = float(np.clip(features[index]@features[reference], 0, 1))
    textual = len(a & b)/math.sqrt(len(a)*len(b)) if a and b else 0.0
    return float(np.clip((visual+textual)/math.sqrt((1+bool(a))*(1+bool(b))),0,1))


def counterfactual_proposal(pool, features, parent, states, soft):
    """接口：固定父参照复算提议，不写模型输入；只判断简单修补是否值得推进。"""
    original = {i for r in parent['regions'] for i in r['prefix'][:parent['region_budgets'][str(r['region_id'])]]}
    proposals = []
    # 步骤1：保留同一末位/名额，禁止额外借位或按已知见证强制选帧。
    for r, region in zip(parent['regions'],pool['plan']['regions']):
        quota = parent['region_budgets'][str(r['region_id'])]
        if quota <= 1: continue
        fixed, removed = r['prefix'][:quota-1], r['prefix'][quota-1]
        L, _, _ = weighted_support(features,r['members'],region['seed_indices'],pool['initial_scores'])
        support = dict(zip(r['members'],map(float,L)))
        def utility(i):
            D = max(companion_similarity(features,states,i,s) if soft else
                float(np.clip(features[i]@features[s],0,1))*text_attenuation(states[i],states[s]) for s in fixed)
            return support[i]*math.sqrt(1-D)
        options = [i for i in r['members'] if i not in original and states[i]]
        if not options: continue
        new = min(options,key=lambda i:(-utility(i),*frame_key(pool['candidates'][i])))
        proposals.append(dict(added_index=new,removed_index=removed,gain=utility(new)-utility(removed),
                              words=states[new],added_frame=pool['candidates'][new],removed_frame=pool['candidates'][removed]))
    positive = [p for p in proposals if p['gain'] > 0]
    # 步骤2：仅保存数值提议，不冻结新算法，不产生答案或事后修改既有输入。
    return min(positive,key=lambda p:(-p['gain'],*frame_key(pool['candidates'][p['added_index']]))) if positive else None


def utility_parts(features, states, index, fixed, support):
    """接口：同一候选/同一已选参照下，输出原视觉D和文字衰减后的D及效用。

    只复算既有分数，不更改参照、候选资格、前缀或配额；差值是代理效用差，
    不是答题价值。单空文字集合会把有字候选的帧对重复度压成0，必须显式展示。
    """
    if not fixed: raise ValueError('Diagnostic reference cannot be empty')
    pairs = []
    # 步骤1：严格复用原逐参照dot顺序，不把max与文字乘法交换。
    for ref in fixed:
        visual = float(np.clip(features[index]@features[ref], 0, 1))
        multiplier = float(text_attenuation(states[index], states[ref]))
        pairs.append(dict(reference_index=ref, reference_tokens=states[ref],
                          visual_similarity=visual, text_multiplier=multiplier,
                          joint_similarity=visual*multiplier))
    visual_D = max(p['visual_similarity'] for p in pairs)
    text_D = max(p['joint_similarity'] for p in pairs)
    # 步骤2：显示惩罚放松幅度；文字方案效用不小于视觉方案是代数性质，不是收益证据。
    if text_D > visual_D + 1e-12: raise ValueError('Unexpected attenuation increase')
    return dict(candidate_index=index, candidate_tokens=states[index], support=float(support),
                visual_redundancy=visual_D, text_redundancy=text_D,
                visual_utility=float(support)*math.sqrt(1-visual_D),
                text_utility=float(support)*math.sqrt(1-text_D), pairs=pairs)


def audit(root):
    """接口：固定8题两配置、实际胜出/移除帧及022已知计时图；模型调用为0。"""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with exclusive(root/'audit.lock'):
        sources = {str(SOURCE/'protocol.json'):sha256(SOURCE/'protocol.json')}
        records = []
        # 步骤1：按原冻结顺序读取实际有界提议，粗细源身份和特征映射必须一致。
        for q in PILOT:
            for profile, k in [('qwen32',32), ('llava8',8)]:
                path = SOURCE/'offline'/profile/(q+'.json'); item = read_json(path)
                trace_path = SOURCE/item['trace_file']; trace = read_json(trace_path)
                if sha256(trace_path) != item['trace_sha256']: raise ValueError('Winner trace changed')
                for p, h in item['sources'].items():
                    if sha256(p) != h: raise ValueError('Frozen candidate source changed')
                pp, fp, tp, _ = locations(q,k); pool = read_json(pp)
                parent = read_json(tp); parent = parent['exp07'] if k == 8 else parent
                features = np.load(fp, allow_pickle=False)
                if k == 8: features = features[pool['original_candidate_indices']]
                states = item['states']; winner = trace['winner']
                sources.update(item['sources']); sources.update({str(path):sha256(path),str(trace_path):sha256(trace_path)})
                if winner is None:
                    records.append(dict(question_id=q,profile=profile,changed=False)); continue
                rid = winner['region_id']; region = next(r for r in pool['plan']['regions'] if r['region_id'] == rid)
                members = next(r for r in parent['regions'] if r['region_id'] == rid)['members']
                L, _, _ = weighted_support(features,members,region['seed_indices'],pool['initial_scores'])
                support = dict(zip(members,map(float,L))); fixed = winner['fixed_indices']
                old = utility_parts(features,states,winner['removed_index'],fixed,support[winner['removed_index']])
                new = utility_parts(features,states,winner['added_index'],fixed,support[winner['added_index']])
                if (abs(new['text_utility']-winner['new_utility']) > 1e-8
                        or abs(old['text_utility']-winner['old_utility']) > 1e-8):
                    raise ValueError('Actual utility trace not reproduced')
                rec = dict(question_id=q,profile=profile,changed=True,region_id=rid,
                    old=dict(old,frame=pool['candidates'][old['candidate_index']]),
                    new=dict(new,frame=pool['candidates'][new['candidate_index']]),
                    actual_text_gain=winner['gain'],same_reference_visual_gain=new['visual_utility']-old['visual_utility'])
                # 022-1计时图是独立已知见证，只诊断既有竞争，不能拿它替换自动赢家。
                if q == '022-1' and 25 in members:
                    rec['known_clock_probe'] = dict(utility_parts(features,states,25,fixed,support[25]),
                        eligible_for_replacement=25 not in trace['parent_selected_indices'])
                confirmed_path = CONFIRMED/'offline'/profile/(q+'.json')
                confirmed = read_json(confirmed_path)
                for p,h in confirmed['sources'].items():
                    if sha256(p) != h: raise ValueError('Confirmed-word source changed')
                sources.update(confirmed['sources']); sources[str(confirmed_path)] = sha256(confirmed_path)
                rec['confirmed_words_only_proposal'] = counterfactual_proposal(pool,features,parent,confirmed['states'],False)
                rec['soft_companion_proposal'] = counterfactual_proposal(pool,features,parent,confirmed['states'],True)
                records.append(rec)
        # 步骤2：保存全体结果和来源；不将数学放松幅度包装成精度或语义有效性。
        value = dict(records=records,sources=sources,code_sha256=sha256(Path(__file__)),
            code_dependencies={str(ROOT/'src/videoqa_methods'/n):sha256(ROOT/'src/videoqa_methods'/n)
                for n in ('weighted_seed_support.py','text_state_redundancy.py','density_selection.py')},
            new_model_calls=0,historical_selection_writes=0,
            limitation='Actual utility decomposition plus unexecuted proposals; not algorithm accuracy or semantic evidence truth')
        out = root/'utility_audit.json'
        if out.exists() and read_json(out) != value: raise ValueError('Read-only audit changed on recovery')
        if not out.exists(): durable(out,value)
        return value


def main():
    """接口：独立零调用分析入口，不修改任何冻结选择器或实验快照。"""
    p = argparse.ArgumentParser(); p.add_argument('--run-id', required=True); a = p.parse_args()
    if not a.run_id.replace('_','').isalnum(): p.error('Invalid run ID')
    audit(ROOT/'outputs/analysis'/a.run_id)


if __name__ == '__main__': main()
