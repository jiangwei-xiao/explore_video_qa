"""Post-run statistics; analysis provenance is separate from frozen inference code."""
from collections import Counter
import csv
import datetime
import shutil
from pathlib import Path
import numpy as np

from videoqa_runtime.common import ROOT, read_json, write_json, sha256
from videoqa_runtime.baseline_report import wilson, distribution
from .records import VARIANTS, result_path, validate_group
from .algorithm import membership

METHODS = ('uniform', 'topk', 'A', 'B', 'C', 'D')
COMPARISONS = [('C','uniform'),('C','topk'),('A','uniform'),('A','topk'),('B','A'),('C','B'),('D','topk'),('C','D')]


def load_all(run):
    """统计输入接口：读取两组历史基线与四组新结果，先核验50题完整配对和D的C来源身份。"""
    run = Path(run)
    rows = read_json(ROOT / 'data/manifests/video_mme_development.json')['rows']
    records = {m: {} for m in METHODS}
    for method in METHODS:
        directory = run / ('baseline_reference/results' if method in ('uniform','topk') else 'results') / method
        paths = list(directory.glob('*.json'))
        if len(paths) != 50:
            raise ValueError(f'{method}: expected 50 results, got {len(paths)}')
        for path in paths:
            record = read_json(path)
            records[method][record['question_id']] = record
    for row in rows:
        qid = row['question_id']
        validate_group({m: records[m][qid] for m in VARIANTS}, row, sha256(run/'protocol.json'), records['topk'][qid])
        assert sha256(result_path(run,'C',qid)) == records['D'][qid]['donor_result_sha256']
    return rows, records


def predicted(record):
    """兼容历史与本轮记录，返回已经解析的答案，不重新调用模型。"""
    return record.get('answer', record.get('result', {}))['parsed_answer']


def chosen(record):
    """兼容记录结构，返回实际进入问答模型的16个源帧记录。"""
    return record.get('pool', record.get('selection'))['selected_frames']


def pair_stats(rows, records, left, right):
    """统计接口：按时长分层进行固定种子的配对bootstrap，同时记录四格得失与按冻结题序选出的前2个正负案例。"""
    rng = np.random.default_rng(2027)
    sampled = np.zeros(10000)
    for layer in ('short','medium','long'):
        group = [r for r in rows if r['stratum'] == layer]
        values = np.array([int(records[left][r['question_id']]['correct']) - int(records[right][r['question_id']]['correct']) for r in group])
        indices = rng.integers(0,len(values),size=(10000,len(values)))
        sampled += values[indices].sum(axis=1)
    cells = {name: [] for name in ('both_correct','improved','regressed','both_wrong')}
    for row in rows:
        qid = row['question_id']; a,b = records[left][qid]['correct'], records[right][qid]['correct']
        cells['both_correct' if a and b else 'improved' if a else 'regressed' if b else 'both_wrong'].append(qid)
    difference = 100 * (len(cells['improved']) - len(cells['regressed'])) / len(rows)
    return dict(left=left,right=right,difference_pp=difference,
                bootstrap_95_pp=np.percentile(sampled/len(rows)*100,[2.5,97.5]).tolist(),
                paired_cells=cells, detailed_cases=dict(improved=cells['improved'][:2],regressed=cells['regressed'][:2]))


def summarize(run):
    """核心分析接口：生成六组正确率、成本、机制诊断、固定案例索引及仅供下一轮讨论的优化提案；不运行新的问答或分类。"""
    run = Path(run)
    # 步骤1：先验证200条新结果与历史对照的完整性，再计算正确率及分时长性能。
    rows, records = load_all(run)
    methods = {}
    for method in METHODS:
        values = list(records[method].values())
        groups = {}
        for layer in ('all','short','medium','long'):
            group = values if layer == 'all' else [r for r in values if r['stratum'] == layer]
            count = sum(r['correct'] for r in group)
            groups[layer] = dict(n=len(group),correct=count,accuracy=count/len(group),wilson_95=wilson(count,len(group)),
                                timings={k: distribution([r['timings'][k] for r in group]) for k in group[0]['timings']},
                                memory={k: distribution([r['memory'][k] for r in group]) for k in group[0]['memory']})
        methods[method] = dict(by_stratum=groups,parse_failures=[r['question_id'] for r in values if predicted(r) is None],
                               performance_source='historical baseline' if method in ('uniform','topk') else 'donor-attributed diagnostic' if method=='D' else 'current direct measurement')
    # 步骤2：保留所有预设比较，使用同一分层配对bootstrap规则，避免只选有利对比。
    comparisons = {f'{a}-{b}': pair_stats(rows,records,a,b) for a,b in COMPARISONS}
    # 步骤3：从已选帧及保存的纯视觉特征重算机制代理指标，不运行新模型调用。
    per_question = []
    for row in rows:
        qid = row['question_id']
        c_pool = records['C'][qid]['pool']
        for method in METHODS:
            record = records[method][qid]
            pool = record['pool'] if method in VARIANTS else c_pool
            positions = {r['source_pts']: i for i,r in enumerate(pool['candidates'])}
            selected = [positions[r['source_pts']] for r in chosen(record)]
            features_source = record if method in VARIANTS else records['C'][qid]
            features = np.load(run / features_source['feature_file'], mmap_mode='r', allow_pickle=False)
            f = features[selected]
            groups = membership(pool['candidates'],pool['segments'])[selected]
            similarities = np.clip(f @ f.T,0,1)
            duplicate = []
            for i,sid in enumerate(groups):
                peers = [j for j,other in enumerate(groups) if i!=j and other==sid]
                duplicate.append(float(similarities[i,peers].max()) if peers else 0.)
            quotas = np.bincount(groups,minlength=len(pool['segments'])).tolist()
            stats = dict(question_id=qid,stratum=row['stratum'],method=method,correct=record['correct'],
                         prediction=predicted(record),scope=record['pool']['scope']['label'] if method in ('B','C') else None,
                         segments=len(pool['segments']),covered_segments=len(set(groups.tolist())),
                         zero_quota_fraction=sum(x==0 for x in quotas)/len(quotas),selected_quotas=quotas,
                         mean_selected_relevance=float(np.mean([pool['scores'][i] for i in selected])),
                         mean_selected_max_same_segment_cosine=float(np.mean(duplicate)),
                         fine_candidates=len(pool['candidates'])-pool['initial_count'] if method in ('C','D') else 0,
                         new_selected=sum(i>=pool['initial_count'] for i in selected) if method in ('C','D') else 0,
                         hotspot_count=sum(h['selected'] for h in pool['hotspot_trace']) if method=='C' else 0)
            per_question.append(stats)
    mechanisms = {}
    for m in METHODS:
        selected = [r for r in per_question if r['method']==m]
        mechanisms[m] = {key: distribution([r[key] for r in selected]) for key in (
            'segments','covered_segments','zero_quota_fraction','mean_selected_relevance',
            'mean_selected_max_same_segment_cosine','fine_candidates','new_selected','hotspot_count')}
    scope = {m:dict(counts=dict(Counter(records[m][r['question_id']]['pool']['scope']['label'] for r in rows)),
                       fallbacks=sum(records[m][r['question_id']]['pool']['scope']['fallback'] for r in rows)) for m in ('B','C')}
    overlaps = {}
    for left,right in [('A','B'),('B','C'),('C','D')]:
        values = []
        for row in rows:
            a,b = [{r['source_pts'] for r in chosen(records[m][row['question_id']])} for m in (left,right)]
            values.append(len(a & b)/len(a | b))
        overlaps[f'{left}-{right}'] = dict(distribution=distribution(values),by_question={r['question_id']:v for r,v in zip(rows,values)})
    cases = [r['question_id'] for r in rows if len({predicted(records[m][r['question_id']]) for m in METHODS})>1]
    attempts = [read_json(p) for p in (run/'attempts').glob('*/*.json')]
    # 步骤4：分类和问答分开计数；C候选池被D引用不构成新的分类或补查。
    calls = dict(final_answers_started=sum(bool(a.get('answer_state',{}).get('started')) for a in attempts),
                 final_answers_returned=sum(bool(a.get('answer_state',{}).get('returned')) for a in attempts),
                 scope_started=sum(bool(a.get('scope_state',{}).get('started')) for a in attempts),
                 scope_returned=sum(bool(a.get('scope_state',{}).get('returned')) for a in attempts),
                 attempts=len(attempts),failed_attempts=sum(a['status']!='completed' for a in attempts))
    cstats = [r for r in per_question if r['method']=='C']
    refinement = dict(triggered_questions=sum(r['hotspot_count']>0 for r in cstats),
                      questions_with_new_frames=sum(r['fine_candidates']>0 for r in cstats),
                      added_candidates=sum(r['fine_candidates'] for r in cstats),
                      new_frames_selected=sum(r['new_selected'] for r in cstats),
                      questions_with_new_selected=sum(r['new_selected']>0 for r in cstats))
    output = dict(run_id=run.name,methods=methods,comparisons=comparisons,scope=scope,mechanisms=mechanisms,
                  overlaps=overlaps,refinement=refinement,per_question_mechanisms=per_question,
                  changed_prediction_cases=cases,calls=calls,execution=read_json(run/'execution.json'),
                  exploratory=True,baseline_latency_comparison='descriptive across different batches',
                  result_sha256={str(p.relative_to(run)):sha256(p) for v in VARIANTS for p in sorted((run/'results'/v).glob('*.json'))})
    # 步骤5：输出机器可读统计、逐题CSV和固定顺序案例链接。
    write_json(run/'summary.json',output)
    with (run/'per_question.csv').open('w',newline='') as f:
        columns=['question_id','stratum','method','prediction','correct','scope','segments','covered_segments','zero_quota_fraction',
                 'fine_candidates','new_selected','selection_core_seconds','selection_total_seconds','observed_e2e_seconds','attributed_diagnostic_e2e_seconds','timing_kind']
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader()
        for item in per_question:
            record=records[item['method']][item['question_id']]
            out={k:item[k] for k in columns if k in item}
            out.update(selection_core_seconds=record['timings']['selection_core_seconds'],
                       selection_total_seconds=record['timings']['selection_total_seconds'],
                       observed_e2e_seconds=record['timings']['end_to_end_seconds'],
                       attributed_diagnostic_e2e_seconds=record['timings'].get('diagnostic_attributed_e2e_seconds',''),
                       timing_kind=methods[item['method']]['performance_source'])
            w.writerow(out)
    lines=[f'# 方法首版实验：{run.name}','','复用两组历史基线；A/B/C/D各50题新测，方法参数未按结果调整。', '',
           '| 方法 | 正确数 | 准确率 | 95% Wilson | 选帧核心均值(s) | E2E/诊断归因均值(s) | 口径 |',
           '|---|---:|---:|---|---:|---:|---|']
    for m in METHODS:
        a=methods[m]['by_stratum']['all']; ts=a['timings']; ci=a['wilson_95']
        e=ts['diagnostic_attributed_e2e_seconds']['mean'] if m=='D' else ts['end_to_end_seconds']['mean']
        lines.append(f'| {m} | {a["correct"]}/50 | {a["accuracy"]:.1%} | {ci[0]:.1%}–{ci[1]:.1%} | {ts["selection_core_seconds"]["mean"]:.6f} | {e:.3f} | {methods[m]["performance_source"]} |')
    lines += ['', 'D的核心时间是使用C候选池后的排名增量；其归因总成本加入C产生候选池的前序实际成本，不包含C的最终重选和回答，不是独立算法直接E2E。', '',
              '| 对比（左−右） | 差值(pp) | 分层配对95%区间(pp) | 改善/退化题数 |', '|---|---:|---|---|']
    for name,pair in comparisons.items():
        cells=pair['paired_cells']; ci=pair['bootstrap_95_pp']
        lines.append(f'| {name} | {pair["difference_pp"]:+.1f} | [{ci[0]:.1f}, {ci[1]:.1f}] | {len(cells["improved"])}/{len(cells["regressed"])} |')
    lines += ['', '上述比较为探索性分析，不从多个比较中只挑有利结果宣称确认性优势；50题不代表完整评测集。', '',
              '## 机制观察', '', f'- 分类：{scope}。',
              f'- 补查触发：{refinement["triggered_questions"]}/50题；新增候选{refinement["added_candidates"]}帧，进入最终输入{refinement["new_frames_selected"]}帧，涉及{refinement["questions_with_new_selected"]}题。',
              '- 选帧Jaccard均值：'+', '.join(f'{k}={v["distribution"]["mean"]:.3f}' for k,v in overlaps.items())+'。', '',
              '分段数量、配额、覆盖、相关性与重复程度均为代理指标，不能直接解释为真实事件或证据充分性。分时长准确率、全部计时分位数和显存见summary.json。', '',
              '## 固定规则案例', '', '| 对比 | 前2个改善案例 | 前2个退化案例 |', '|---|---|---|']
    for name,pair in comparisons.items():
        cells=pair['detailed_cases']
        links=lambda ids: ', '.join(f'[{qid}](cases/{qid}/index.html)' for qid in ids) or '无'
        lines.append(f'| {name} | {links(cells["improved"])} | {links(cells["regressed"])} |')
    # 步骤6：依据已冻结结果排序优化提案；只写建议，不自动修改参数或追加实验。
    recommendations=[]
    ba=comparisons['B-A']['difference_pp']; cb=comparisons['C-B']['difference_pp']; cd=comparisons['C-D']['difference_pp']
    recommendations.append(dict(priority=1 if ba<=0 else 3,topic='λ作用解耦',evidence=f'B−A={ba:+.1f}pp；类别分布{scope["B"]["counts"]}',
        proposal='下一轮讨论分离覆盖权重与重复惩罚的相对强度，保持其余模块不变；不把本轮B−A归因于纯覆盖作用。'))
    recommendations.append(dict(priority=1 if cb<=0 else 3,topic='补查收益判断',evidence=f'C−B={cb:+.1f}pp；触发{refinement["triggered_questions"]}题、采用新帧{refinement["questions_with_new_selected"]}题',
        proposal='结合触发、换帧和问答变化区分未触发、未采用和采用无效；据此讨论未覆盖指标或重选准则，不直接扩大FPS或搜索参数。'))
    recommendations.append(dict(priority=2 if cd<=0 else 4,topic='固定配额与片段代表性',evidence=f'C−D={cd:+.1f}pp；C零配额比例均值{mechanisms["C"]["zero_quota_fraction"]["mean"]:.3f}',
        proposal='检查D使用而C遗漏的区域及孤立高分峰；据案例讨论有限全局回补或稳健片段代表分数，一次只改一个机制。'))
    recommendations.sort(key=lambda r:r['priority'])
    write_json(run/'optimization_proposals.json',dict(status='proposals_only_not_executed',items=recommendations))
    lines += ['', '## 下一轮优化提案（未执行）', '']
    for item in recommendations:
        lines += [f'- {item["topic"]}：{item["evidence"]}。{item["proposal"]}', '']
    lines += [f'调用记录：{calls}。实际后台墙钟{output["execution"]["wall_seconds"]/60:.2f}分钟。', '',
              '首轮结果保持冻结；优化建议需要新一轮明确对照再验证，本轮不追加调参或模型调用。', '']
    (run/'report.md').write_text('\n'.join(lines))
    # 步骤7：另存分析代码快照，不改变已经冻结的方法执行指纹。
    for path in [Path(__file__),ROOT/'src/videoqa_methods/visualize.py']:
        target=run/'analysis_code_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    write_json(run/'analysis_manifest.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        analysis_code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/videoqa_methods/visualize.py']},
        summary_sha256=sha256(run/'summary.json')))
    print('Accuracies:',{m:methods[m]['by_stratum']['all']['accuracy'] for m in METHODS},flush=True)
    return output
