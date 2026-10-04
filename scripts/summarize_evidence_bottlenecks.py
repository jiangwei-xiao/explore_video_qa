"""汇总冻结见证的最低配额代价；只解释结构，不推断证据充分性或生成新答案。"""
import argparse
import hashlib
import json
from pathlib import Path


def summarize_fact(fact):
    """输入同一事实的已列见证路径，输出命中状态和单事实最低配额增量。

    不同事实的增量不能相加：它们可能共享区域、见证或争夺同一预算。
    未命中只表示未选到目录中的帧，不表示替代粗帧或细帧没有证据。
    """
    # 步骤1：校验目录命中与逐见证记录一致，避免沿用过期摘要。
    paths = [p['exp12'] for p in fact['paths']]
    if not paths or any(p['selected'] for p in paths) != fact['current_hit']:
        raise ValueError('事实命中与逐见证路径不一致')
    # 步骤2：仅在锚点下界约束下仍可到达的位置计算代价，不改变局部次序。
    reachable = [p for p in paths if p['prefix_position'] is not None
                 and p['prefix_position'] <= p['max_feasible_quota']]
    extra = min((max(0, p['prefix_position'] - p['quota']) for p in reachable), default=None)
    state = ('selected' if fact['current_hit'] else
             'quota_reachable' if extra is not None else 'order_or_representation_required')
    return dict(question_id=fact['question_id'], fact_id=fact['fact_id'],
                description=fact['description'], state=state,
                minimum_extra_slots=extra, witnesses=paths)


def build(source, output):
    """读取冻结路径并原子保存派生分析；恢复必须逐字一致，禁止覆盖变化结果。"""
    # 步骤1：绑定输入字节及分析代码，固定75事实分母，不根据答案筛选。
    raw = source.read_bytes()
    data = json.loads(raw)
    facts = [summarize_fact(f) for f in data['facts']]
    if len(facts) != 75 or len({f['question_id'] for f in facts}) != 42:
        raise ValueError('冻结分母不符')
    # 步骤2：列出单事实配额代价与理论不可达项，不将可达性解释为有效改进。
    result = dict(source_sha256=hashlib.sha256(raw).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  new_model_calls=0, facts=facts,
                  counts={s: sum(f['state'] == s for f in facts) for s in
                          ['selected', 'quota_reachable', 'order_or_representation_required']},
                  limitation='单事实结构下界；不是联合预算可行解，也不是语义充分性或准确率')
    # 步骤3：完成项只核对不重写；派生文件独立保存，源结果不修改。
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode()
    if output.exists():
        if output.read_bytes() != encoded:
            raise ValueError('恢复结果或指纹变化')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp')
        temporary.write_bytes(encoded)
        temporary.replace(output)
    print(json.dumps(result['counts'], ensure_ascii=False))
    for f in facts:
        if f['state'] == 'quota_reachable':
            print(f['question_id'], f['fact_id'], f['minimum_extra_slots'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
