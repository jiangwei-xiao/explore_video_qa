"""完整选项词覆盖：只奖励可指认的任务文字首次见证，不奖励任意OCR变化。"""
import math
import numpy as np
from videoqa_methods.text_state_redundancy import normalize_text_state
from videoqa_methods.density_selection import frame_key, validate_candidates
from videoqa_methods.weighted_seed_support import weighted_support


def option_word_groups(question, options):
    """接口：公开原题/四选项→完整词集合组；无gold，不改字、不猜数字/词义。

    词序不同但词集合相同合并为同一文字见证组，不能据此判断时序答案。
    纯数字及完全已在问题中的词组不作为新文字证据；不引入可调长度阈值。
    """
    if len(options) != 4 or not isinstance(question, str):
        raise ValueError('Original question and four ordered options required')
    question_words = set(normalize_text_state([dict(text=question, score=1.0)]))
    groups = []
    # 步骤1：去掉原选项字母前缀，仅用完整正文，不挑正确选项或生成替代描述。
    for letter, option in zip('ABCD', options):
        if not option.startswith(letter + '.'):
            raise ValueError('Original option order differs')
        words = set(normalize_text_state([dict(text=option[2:].strip(), score=1.0)]))
        if not words or not any(any(char.isalpha() for char in word) for word in words) or words <= question_words:
            continue
        canonical = tuple(sorted(words))
        # 步骤2：重复或仅顺序不同的答案保持原计分，见证组不重复计算预算奖励。
        if canonical not in groups:
            groups.append(canonical)
    return groups


def literal_matches(states, groups):
    """接口：冻结OCR词集合→覆盖完整选项词组的组号；不做词干、纠错或模糊匹配。"""
    return [set(i for i, group in enumerate(groups) if set(group) <= set(state)) for state in states]


def select_literal_coverage(pool, features, parent, matches, budget):
    """接口：原池/视觉特征/配额/文字见证组→选择与轨迹；未覆盖组降低局部重复度。

    每个单元保留父锚点、原L及配额。一次覆盖后不继续给同组新字奖励。
    无匹配时原公式等价；文字见证不自动说明与问句的关系或答案是否正确。
    """
    candidates = pool['candidates']; validate_candidates(candidates, pool['video'])
    f = np.asarray(features, dtype=np.float32)
    if len(matches) != len(candidates) or f.ndim != 2 or len(f) != len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError('OCR and normalized visual source populations differ')
    chosen_all = []; traces = []; owned = []
    # 步骤1：只在原单元内覆盖文字组，不跨段删除同名事件，也不重算覆盖DP。
    for region, plan in zip(parent['regions'], pool['plan']['regions']):
        members = region['members']; owned += members; anchor = plan['anchor_index']
        quota = parent['region_budgets'][str(region['region_id'])]
        if region['region_id'] != plan['region_id'] or anchor not in members or not 1 <= quota <= len(members):
            raise ValueError('Parent membership or quota mismatch')
        support, _, _ = weighted_support(f, members, plan['seed_indices'], pool['initial_scores'])
        chosen = [anchor]; covered = set(matches[anchor]); steps = []
        redundancy = np.clip(f[members] @ f[anchor], 0, 1)
        # 步骤2：新组占该候选组的比例是派生量；没有新组或没有文字时沿用原视觉D。
        while len(chosen) < quota:
            proposals = []
            for position, i in enumerate(members):
                if i in chosen:
                    continue
                new_groups = set(matches[i]) - covered
                novelty = len(new_groups) / len(matches[i]) if matches[i] else 0.0
                visual_d = float(redundancy[position]); d = visual_d * (1 - novelty)
                proposals.append(dict(candidate_index=i, L=float(support[position]),
                    visual_D=visual_d, literal_novelty=novelty, adjusted_D=d,
                    newly_covered_groups=sorted(new_groups), score=float(support[position]) * math.sqrt(1 - d)))
            winner = min(proposals, key=lambda value: (-value['score'], *frame_key(candidates[value['candidate_index']])))
            i = winner['candidate_index']; steps.append(dict(selected_before=list(chosen), covered_before=sorted(covered), winner=winner))
            chosen.append(i); covered.update(matches[i])
            redundancy = np.maximum(redundancy, np.clip(f[members] @ f[i], 0, 1))
        chosen_all += chosen; traces.append(dict(region_id=region['region_id'], quota=quota,
            prefix=chosen, final_covered_groups=sorted(covered), steps=steps))
    # 步骤3：全原池、配额、唯一源闭合；返回真实时间顺序，不把OCR词写入问答提示。
    expected = sum(parent['region_budgets'].values())
    if sorted(owned) != list(range(len(candidates))) or len(chosen_all) != expected or len(set(chosen_all)) != expected or expected > budget:
        raise ValueError('Fixed budget or candidate ownership mismatch')
    selected = sorted(chosen_all, key=lambda i: frame_key(candidates[i]))
    return selected, dict(regions=traces, region_budgets=parent['region_budgets'],
        selected_indices=selected, rule='L * sqrt(1 - visual_D * (1 - unseen-option-group fraction)); original quotas')
