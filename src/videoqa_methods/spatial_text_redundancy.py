"""同位文字状态冗余：只修正有空间对应的文字变化，不奖励单侧检测或任意新字。"""
from functools import lru_cache
import math
import unicodedata
import numpy as np
from videoqa_methods.density_selection import frame_key, validate_candidates
from videoqa_methods.weighted_seed_support import weighted_support


def layout_state(regions, rgb_shape):
    """接口：冻结OCR原框/原字/原尺寸→归一化矩形及字符，沿用旧0.5识别阈值。

    不纠正O/0等疑似读错，不输入问题/选项/答案。矩形只代表潜在空间对应，
    不是同一语义字段的保证；旋转文字暂用包围框，并保留原识别的不确定性。
    """
    if len(rgb_shape) != 3 or rgb_shape[2] != 3 or min(rgb_shape[:2]) <= 0:
        raise ValueError('Invalid source RGB geometry')
    height, width = rgb_shape[:2]; result = []
    # 步骤1：可信度规则不重新调参；无字或低分原输出在源账本保留。
    for region in regions:
        confidence = region['score']
        if not math.isfinite(confidence) or not 0 <= confidence <= 1: raise ValueError('Invalid OCR confidence')
        if confidence < 0.5: continue
        text = ' '.join(unicodedata.normalize('NFC', region['text'].replace('’', "'")).casefold().split())
        if not text: continue
        points = np.asarray(region['box'], dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all(): raise ValueError('Invalid OCR polygon')
        x1, y1 = points.min(axis=0); x2, y2 = points.max(axis=0)
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            raise ValueError('OCR box outside actual source canvas')
        result.append(((float(x1 / width), float(y1 / height), float(x2 / width), float(y2 / height)), text))
    # 步骤2：只统一画布/字符格式，不生成新值或把识别分数解释为正确概率。
    return tuple(result)


def box_iou(a, b):
    """接口：两个归一化矩形的交并比；共享端点但没有面积时对应强度为0。"""
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)


@lru_cache(maxsize=65536)
def character_change(left, right):
    """接口：单位插入/删除/替换的编辑距离÷较长字符数，保留字符顺序，不当作语义距离。"""
    if left == right: return 0.0
    if len(left) < len(right): left, right = right, left
    # 步骤1：滚动行编辑距离，有界内存；同位一字符变动不会直接当全新信息。
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    # 步骤2：归一化只是变化强度代理，不证明读取正确或内容对题目有用。
    return previous[-1] / max(1, len(left))


def layout_change(left, right):
    """接口：先以几何关系确定唯一相互最佳框，再计算对应字符变化。

    不用文字差异选择对应对象，否则相同布局内不同框会产生虚假的自差异。
    空/无重叠/几何对应并列时不奖励；这里没有新增IoU阈值或拟合权重。
    """
    if not left or not right: return 0.0
    # 步骤1：对应关系只由空间确定；并列最佳时承认歧义，不随框输入顺序猜测。
    overlap = np.asarray([[box_iou(a, b) for b, _ in right] for a, _ in left])
    def unique_best(values):
        """接口：非空IoU行/列→唯一正最大索引；零值或并列代表没有可靠对应。"""
        maximum = float(values.max())
        indices = np.flatnonzero(values == maximum)
        return int(indices[0]) if maximum > 0 and len(indices) == 1 else None
    rows = [unique_best(row) for row in overlap]
    cols = [unique_best(column) for column in overlap.T]
    # 步骤2：只消费双向唯一对应；文字变化不再能制造自己偏好的空间匹配。
    best = 0.0
    for i, j in enumerate(rows):
        if j is not None and cols[j] == i:
            best = max(best, float(overlap[i, j]) * character_change(left[i][1], right[j][1]))
    return best


def select_spatial_text(pool, features, parent_trace, states, frame_budget):
    """接口：固定父候选/特征/配额，只以cos×(1−同位文字变化)重算局部前缀。

    L、锚点和配额沿用exp07。每个帧对先修正视觉相似，再对已选集合取最大值；
    空文字检测没有单侧新奇奖励。OCR只用于选择，不加入最终问答提示词。
    """
    candidates = pool['candidates']; validate_candidates(candidates, pool['video'])
    f = np.asarray(features, dtype=np.float32)
    if len(states) != len(candidates) or f.ndim != 2 or len(f) != len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError('Candidate/features/layout identity differs')
    records = []; selected = []
    # 步骤1：固定区域归属、原题支持与预算，不同时改定位、补查或分配目标。
    for original, region in zip(parent_trace['regions'], pool['plan']['regions']):
        ids = original['members']; anchor = region['anchor_index']; quota = parent_trace['region_budgets'][str(region['region_id'])]
        if original['region_id'] != region['region_id'] or not 1 <= quota <= len(ids) or anchor not in ids:
            raise ValueError('Parent allocation mismatch')
        support, _, _ = weighted_support(f, ids, region['seed_indices'], pool['initial_scores'])
        chosen = [anchor]; nearest = np.full(len(ids), anchor, dtype=np.int64)
        def paired(reference):
            """接口：一个新参照→逐候选修正后的相似度，再由外层更新最大重复度。"""
            visual = np.clip(f[ids] @ f[reference], 0, 1)
            changes = np.asarray([layout_change(states[i], states[reference]) for i in ids], dtype=np.float32)
            return visual * (1 - changes)
        redundancy = paired(anchor); steps = []
        # 步骤2：一次只更新新参照，避免构造所有帧的文字全矩阵；同分沿用PTS/编号。
        while len(chosen) < min(frame_budget, len(ids)):
            options = [dict(candidate_index=i, support=float(support[j]), redundancy=float(redundancy[j]),
                redundancy_reference=int(nearest[j]), local_score=float(support[j]) * math.sqrt(1 - float(redundancy[j])))
                for j, i in enumerate(ids) if i not in chosen]
            winner = min(options, key=lambda p: (-p['local_score'], *frame_key(candidates[p['candidate_index']])))
            index = winner['candidate_index']; chosen.append(index); steps.append(winner)
            update = paired(index); improved = update > redundancy; nearest[improved] = index
            redundancy = np.maximum(redundancy, update)
        selected += chosen[:quota]
        records.append(dict(original, prefix=chosen, local_steps=steps, parent_prefix=original['prefix'], quota=quota,
            coverage_curve=None, coverage_not_used='Fixed parent allocation; no budget reoptimization'))
    # 步骤3：闭合预算、保护锚点与唯一源帧，恢复精确时间顺序。
    selected = sorted(selected, key=lambda i: frame_key(candidates[i]))
    target = min(frame_budget, len(candidates))
    if len(selected) != target or len(set(selected)) != target: raise ValueError('Budget mismatch')
    if sorted(i for r in records for i in r['members']) != list(range(len(candidates))):
        raise ValueError('Candidate ownership mismatch')
    return selected, dict(regions=records, region_budgets=parent_trace['region_budgets'],
        protected_indices=[r['anchor_index'] for r in pool['plan']['regions']], selected_indices=selected,
        rule='exp07 fixed quotas; pairwise visual cosine * (1 - max bboxIoU * normalized edit distance)',
        limitation='Correspondence and OCR are proxies; cold-start text, misread glyphs or unrelated overlays may remain')
