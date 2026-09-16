"""将人工核实的可见事实映射到真实输入组合，不自动推断整组证据充分性。"""


def map_question(question, outcomes, specification):
    """输入冻结题目、六组输出和人工事实说明，返回逐组PTS见证及B/C保留关系。"""
    frames = question['frames']
    facts = specification['facts']
    result = {'question_id': question['question_id'], 'question': question['question'],
              'reasoning_limits': specification['reasoning_limits'], 'facts': [], 'groups': {}}
    # 步骤1：联合图板编号只用于定位；实际运算使用精确源PTS，禁止跨题复用序号。
    ids = set()
    for fact in facts:
        if fact['id'] in ids:
            raise ValueError('事实编号重复')
        ids.add(fact['id'])
        indices = fact['union_indices']
        if len(set(indices)) != len(indices) or any(type(i) is not int or i < 1 or i > len(frames) for i in indices):
            raise ValueError('事实图板编号无效')
        result['facts'].append({**fact, 'witnesses': [frames[i - 1] for i in indices]})
    # 步骤2：只记录明确见证是否被保留；无见证不能自动标记原片无证据。
    by_method = {}
    for alias, pts in question['groups'].items():
        if len(pts) != 16 or len(set(pts)) != 16:
            raise ValueError('原输入不是16个唯一PTS')
        if not set(pts).issubset({f['source_pts'] for f in frames}):
            raise ValueError('组内PTS未包含在冻结联合图板')
        selected = set(pts)
        group = {**outcomes[alias], 'facts': {}, 'sufficiency': 'not_inferred_from_witness_counts'}
        for fact in result['facts']:
            witnesses = [f for f in fact['witnesses'] if f['source_pts'] in selected]
            group['facts'][fact['id']] = {'count': len(witnesses), 'witnesses': witnesses,
                'state': 'witness_present' if witnesses else 'no_listed_witness_in_input'}
        result['groups'][alias] = group
        by_method[group['method']] = group
    # 步骤3：同一事实的多帧视为可替代见证；换掉一帧但仍留同事实，不计为见证完全丢失。
    result['BC_fact_transitions'] = []
    if 'B' in by_method and 'C' in by_method:
        for fact in result['facts']:
            b = by_method['B']['facts'][fact['id']]['count']
            c = by_method['C']['facts'][fact['id']]['count']
            state = ('retained' if c else 'listed_witness_lost') if b else ('listed_witness_added' if c else 'not_witnessed_in_either')
            result['BC_fact_transitions'].append({'fact_id': fact['id'], 'B_count': b, 'C_count': c, 'state': state,
                'warning': '仅对人工明确列出的这一事实及见证成立；不自动等于必要证据损失或答案因果解释。'})
    return result
