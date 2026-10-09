"""同位文字试点只读复核：追踪真实竞争字段，人工计时器擦除仅作诊断。"""
import argparse
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_full.state import durable, utc
from videoqa_runtime.common import read_json, sha256
from videoqa_methods.spatial_text_redundancy import (
    box_iou, character_change, layout_change, select_spatial_text,
)
from run_spatial_text_pilot import verify


def matched_changes(left, right):
    """接口：复算双向唯一几何对应，输出原字、原框及变化贡献，不解释语义。"""
    if not left or not right:
        return []
    # 步骤1：空间匹配不消费文字内容；并列必须与冻结选择器一样放弃。
    matrix = np.asarray([[box_iou(a, b) for b, _ in right] for a, _ in left])
    pairs = []
    for i, row in enumerate(matrix):
        j = int(np.argmax(row))
        value = float(row[j])
        if value <= 0 or np.count_nonzero(row == value) != 1:
            continue
        column = matrix[:, j]
        if int(np.argmax(column)) != i or np.count_nonzero(column == value) != 1:
            continue
        change = character_change(left[i][1], right[j][1])
        pairs.append(dict(left_text=left[i][1], right_text=right[j][1],
            left_box=left[i][0], right_box=right[j][0], iou=value,
            character_change=change, contribution=value * change))
    # 步骤2：诊断说明必须对齐实际选择公式，不另造一个对应算法。
    pairs.sort(key=lambda p: -p['contribution'])
    expected = max((p['contribution'] for p in pairs), default=0.0)
    if not math.isclose(expected, layout_change(left, right), abs_tol=1e-12):
        raise ValueError('Audit correspondence differs from selector')
    return pairs


def audit_steps(pool, features, states, trace):
    """接口：仅查看已保存前缀的真实胜出帧及最大重复参照，验证分数并保留字段。"""
    results = []
    # 步骤1：每区固定父锚点，从轨迹顺序恢复当时已选参照，而非最终集合。
    for region in trace['regions']:
        selected = [region['prefix'][0]]
        for position, step in enumerate(region['local_steps'], 2):
            i = step['candidate_index']; reference = step['redundancy_reference']
            if reference not in selected:
                raise ValueError('Redundancy reference was not selected yet')
            visual = float(np.clip(features[i] @ features[reference], 0, 1))
            pairs = matched_changes(states[i], states[reference])
            change = max((p['contribution'] for p in pairs), default=0.0)
            # 步骤2：核验累计最大D及当前局部分数，浮点比较沿用1e-6容差。
            all_pairs = [float(np.clip(features[i] @ features[j], 0, 1)) *
                (1 - layout_change(states[i], states[j])) for j in selected]
            maximum = max(all_pairs)
            if abs(maximum - step['redundancy']) > 1e-6:
                raise ValueError('Saved cumulative redundancy does not replay')
            score = step['support'] * math.sqrt(1 - step['redundancy'])
            if abs(score - step['local_score']) > 1e-6:
                raise ValueError('Saved local score does not replay')
            results.append(dict(region_id=region['region_id'], prefix_position=position,
                quota=region['quota'], finally_selected=position <= region['quota'],
                candidate_index=i, source_frame=pool['candidates'][i],
                reference_index=reference, reference_frame=pool['candidates'][reference],
                visual_cosine=visual, text_change=change,
                redundancy=step['redundancy'], local_score=step['local_score'],
                dominating_fields=pairs[:3]))
            selected.append(i)
    return results


def remove_labeled_timer(states):
    """接口：仅擦除035-2人工确认的右上计时器；不是运行时过滤或新候选。"""
    masked = []; removed = []
    # 步骤1：这个矩形是已查看原图上的人工标注，不作为跨题阈值或调参规则。
    for i, state in enumerate(states):
        keep = []
        for index, (box, text) in enumerate(state):
            labeled = box[0] >= 0.85 and box[3] <= 0.15
            if labeled and re.fullmatch(r'[0-9]{2}:[0-9]{2}', text):
                removed.append(dict(candidate_index=i, layout_index=index, box=box, text=text))
            else:
                keep.append((box, text))
        masked.append(keep)
    # 步骤2：只有诊断副本变化，原OCR、候选、特征和正式选择一字不改。
    return masked, removed


def execute(root):
    """接口：核验冻结来源并追加字段贡献与诊断，不覆盖原试点结果或调用模型。"""
    root = Path(root); spec = verify(root); reports = []
    output = root / 'semantic_review'; output.mkdir(exist_ok=True)
    total = 0; nonzero = 0
    # 步骤1：全3402源布局自差异核验，修正前产物保持原样，不假装失败没发生。
    for q in spec['pilot_ids']:
        states = read_json(root / 'layouts' / (q + '.json'))['states']
        total += len(states); nonzero += sum(layout_change(s, s) != 0 for s in states)
    if nonzero:
        raise ValueError('Self-layout difference must be zero')
    # 步骤2：逐配置保存真正胜出的候选与当时参照，支持语义复核指向具体字段。
    for key, item in spec['inputs'].items():
        profile, q = key.split(':'); pool = read_json(item['pool'])
        features = np.load(item['features'], allow_pickle=False)[item['global_mapping']]
        states = read_json(root / 'layouts' / (q + '.json'))['states']
        states = [states[i] for i in item['global_mapping']]
        trace = read_json(root / 'traces' / profile / (q + '.json'))
        steps = audit_steps(pool, features, states, trace)
        path = output / 'field_contributions' / profile / (q + '.json')
        durable(path, dict(question_id=q, profile=profile, steps=steps, new_model_calls=0))
        reports.append(dict(question_id=q, profile=profile, path=str(path), sha256=sha256(path)))
    # 步骤3：固定035右上计时器擦除作为归因诊断，不发布第二个选择版本。
    item = spec['inputs']['qwen32:035-2']; pool = read_json(item['pool'])
    features = np.load(item['features'], allow_pickle=False)
    states = read_json(root / 'layouts/035-2.json')['states']
    masked, removed = remove_labeled_timer(states); parent = read_json(item['parent_trace'])
    ids, trace = select_spatial_text(pool, features, parent, masked, 32)
    path = output / '035_timer_erasure_diagnostic.json'
    durable(path, dict(scope='Post-hoc manual overlay erasure, not a deployable variant',
        removed=removed, selected_indices=ids, trace=trace,
        selected_frames=[pool['candidates'][i] for i in ids],
        target56_selected=56 in ids, nearby55_selected=55 in ids, nearby57_selected=57 in ids,
        pair56_62_without_timer=layout_change(masked[56], masked[62]),
        new_model_calls=0, new_QA_calls=0))
    # 步骤4：保存追加产物来源；恢复只允许完全相同的计算，不依赖QA或标准答案。
    summary = dict(source_protocol_sha256=sha256(root / 'protocol.json'),
        audit_code_sha256=sha256(Path(__file__)), self_checks=total, nonzero_self_changes=nonzero,
        reports=reports, timer_diagnostic_sha256=sha256(path), new_model_calls=0, new_QA_calls=0)
    old = output / 'numeric_audit.json'
    if old.exists() and read_json(old) != summary:
        raise ValueError('Completed numeric audit changed')
    durable(old, summary)
    print(dict(completed=True, layouts=total, steps=sum(len(read_json(r['path'])['steps']) for r in reports),
        timer_fields_removed=len(removed), target56_selected=56 in ids,
        nearby55_selected=55 in ids, nearby57_selected=57 in ids, utc=utc()))


def main():
    """接口：明确指定既有分析运行目录，不接收算法参数或生成额度。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    execute(ROOT / 'outputs/analysis' / args.run_id)


if __name__ == '__main__':
    main()
