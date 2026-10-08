"""exp07跨模型与预算共同失败统计；官方任务标签不是机制归因。"""
from collections import Counter

PROFILES=('qwen32','qwen16','llava8','llava16')


def summarize_failures(data, records):
    """接口：完整题目与只读预测→四配置共同失败及带分母的分类统计。

    不以共同失败筛选新测试集，不输出按答案路由的方法或oracle准确率。
    """
    ids=[r['question_id'] for r in data]
    if len(ids)!=len(set(ids)) or set(ids)!=set(records):raise ValueError('Failure population not paired')
    # 步骤1：同题连接四工作点，逐模型/预算而非跨题比较正确性。
    common=[q for q in ids if all(not records[q]['conditions'][p]['correct'] for p in PROFILES)]
    common_set=set(common)
    patterns=Counter(''.join('1' if records[q]['conditions'][p]['correct'] else '0' for p in PROFILES) for q in ids)
    groups={}
    # 步骤2：按官方类别报告共同错数量与该类别完整题数，不用数量替代错误率。
    for key in ('stratum','task_type','domain'):
        grouped={}
        for label in sorted({r[key] for r in data}):
            members=[r for r in data if r[key]==label];wrong=sum(r['question_id'] in common_set for r in members)
            grouped[label]=dict(total=len(members),common_wrong=wrong,common_wrong_rate=wrong/len(members),
                per_profile_wrong={p:sum(not records[r['question_id']]['conditions'][p]['correct'] for r in members) for p in PROFILES})
        groups[key]=grouped
    any_baseline=[q for q in common if any(c['correct'] for p in PROFILES for c in records[q]['baselines'][p].values())]
    all_baseline_wrong=[q for q in common if q not in set(any_baseline)]
    # 步骤3：基线能答对和所有输入都错分开；前者仍不自动证明exp07丢证据。
    return dict(questions=len(ids),profiles=list(PROFILES),common_wrong=len(common),common_wrong_ids=common,
                common_wrong_video_count=len({r['video_id'] for r in data if r['question_id'] in common_set}),
                patterns=dict(patterns),categories=groups,baseline_can_answer_ids=any_baseline,
                all_eight_baselines_wrong_ids=all_baseline_wrong,
                limitations=['Model and budget change together in some comparisons',
                             'Official task types are not visual evidence layout labels',
                             'Incorrect answer does not prove frame-selection failure'])
