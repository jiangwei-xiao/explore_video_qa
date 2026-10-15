"""双数据集复盘：零模型、冻结抽样、路径追踪、格式敏感性统计和有界复核。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc, digest
from videoqa_audit.framework_contraction import BASE, RUN8, RUN16, RUN32, POOL16
from videoqa_audit.cross_dataset_review import (
    METHODS, TASKS, PERIOD_POLICY, normalize_option_period, freeze_balanced_subset,
    audit_selection_paths, cluster_report)
from videoqa_audit.coverage_objective_review import frame_identity, trace_witness_path

ML = ROOT/'outputs/full_videoqa/mlvu_dev__S1__llava8_qwen32__four_methods__20261014_r1'
Q12 = ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
S2 = ROOT/'outputs/full_videoqa/scorer_comparison__blip2__qwen32_llava8__20261014_r1'
FACTS = ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
FILES = ('scripts/analyze_cross_dataset_review.py', 'src/videoqa_audit/cross_dataset_review.py',
         'src/videoqa_audit/coverage_objective_review.py', 'src/videoqa_methods/prefix_coverage_budget.py',
         'src/videoqa_methods/relevance_coverage_budget.py', 'src/videoqa_methods/weighted_seed_support.py')


class Reader:
    """接口：每次实际读取绑定字节摘要；恢复重新核查来源与新产物，不修改历史目录。"""
    def __init__(self):
        self.sources = {}

    def read(self, path):
        path = Path(path)
        raw = path.read_bytes()
        value = hashlib.sha256(raw).hexdigest()
        previous = self.sources.setdefault(str(path), value)
        if value != previous:
            raise ValueError('Source changed during audit: '+str(path))
        return json.loads(raw)

    def bind(self, path):
        path = Path(path)
        value = sha256(path)
        if str(path) in self.sources and self.sources[str(path)] != value:
            raise ValueError('Bound source changed: '+str(path))
        self.sources[str(path)] = value
        return path


def publish(path, value):
    """接口：相同派生记录可恢复；内容不一致时停止，不静默覆盖已冻结统计或清单。"""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen analysis output differs: '+str(path))
    else:
        durable(path, value)


def result_path(dataset, profile, method, q):
    """接口：只解析唯一历史来源，开发导入和同帧引用保持原来源，不生成新答案。"""
    if dataset == 'mlvu':
        base = ML/'qa'/profile
    elif profile in ('qwen16', 'llava16'):
        base = RUN16/profile if method == METHODS[2] else POOL16/profile
    elif profile == 'llava8':
        base = RUN8
    else:
        base = {METHODS[0]: BASE, METHODS[1]: BASE, METHODS[2]: RUN32, METHODS[3]: Q12}[method]
    paths = [base/'results'/method/(q+'.json'), base/'imported/results'/method/(q+'.json')]
    matches = [p for p in paths if p.exists()]
    if len(matches) != 1:
        raise ValueError('Missing/ambiguous result: '+str(paths))
    return matches[0]


def pool_pair(dataset, profile, q, reader):
    """接口：同K两版读取同池、共同局部前缀；不重建评分、补查或选择。"""
    if dataset == 'mlvu':
        path = ML/'pools'/profile/(q+'.json')
        pool = reader.read(path)
        traces = reader.read(ML/'traces'/profile/(q+'.json'))
        left, right = traces['exp07'], traces['exp12']
        features = ML/'features'/profile/(q+'.npy')
    elif profile == 'llava8':
        path = RUN8/'inputs/pools'/(q+'.json')
        pool = reader.read(path)
        traces = reader.read(RUN8/'inputs/traces'/(q+'.json'))
        left, right = traces['exp07'], traces['exp12']
        features = BASE/'features'/(q+'.npy')
    else:
        path = BASE/'pools'/(q+'.json')
        pool = reader.read(path)
        left = reader.read(RUN32/'traces'/(q+'.json'))
        right = reader.read(Q12/'traces'/(q+'.json'))
        features = BASE/'features'/(q+'.npy')
    right = dict(right, regions=[dict(b, members=a['members']) for a, b in zip(left['regions'], right['regions'])])
    return pool, left, right, path, features


def answer_record(row, item, dataset):
    """接口：计分在历史输出返回后执行；标准答案仅在此比较，绝不传给路径或抽样函数。"""
    if item['question_id'] != row['question_id'] or item['video_id'] != row['video_id']:
        raise ValueError('Question/video identity differs')
    answer = item['answer']
    gold = 'ABCD'[row['answer_index']]
    native = answer['parsed_answer']
    if bool(item['correct']) != (native == gold):
        raise ValueError('Stored/native correctness differs')
    if dataset == 'videomme' and (item['question'] != row['question'] or item['options'] != row['options']):
        raise ValueError('Question/options differ')
    if dataset == 'mlvu' and answer['framework_metric']['question_id'].strip() != row['qa_question'].strip():
        raise ValueError('MLVU task text differs')
    normalized = normalize_option_period(answer['raw_output'], native) if dataset == 'mlvu' else native
    return dict(native_prediction=native, prediction=normalized, raw_output=answer['raw_output'],
                native_correct=bool(item['correct']), correct=normalized == gold,
                period_changed=normalized != native,
                shared=item.get('shared', False), representative=item.get('representative', item.get('method')),
                imported=item.get('imported', False))


def summarize_structure(records):
    """接口：聚合结构使用量并按任务分组；不把Top-K帧移除自动标成证据损失。"""
    groups = {}
    for record in records:
        for label in ('all', 'task:'+record['task_type'], 'stratum:'+record['stratum']):
            g = groups.setdefault(label, dict(questions=0, states={'exp07': Counter(), 'exp12': Counter()},
                 input_same=0, transferred_slots=0, anchor_fraction_sum=0., topk_correct_both_wrong=0,
                 both_correct_topk_wrong=0, exp07_only=0, exp12_only=0))
            g['questions'] += 1
            structure, answers = record['structure'], record['answers']
            g['input_same'] += structure['selected_indices']['exp07'] == structure['selected_indices']['exp12']
            g['transferred_slots'] += structure['transferred_slots']
            g['anchor_fraction_sum'] += structure['anchor_fraction']
            a, b, t = [answers[m]['correct'] for m in (METHODS[2], METHODS[3], METHODS[1])]
            g['topk_correct_both_wrong'] += bool(t and not a and not b)
            g['both_correct_topk_wrong'] += bool(a and b and not t)
            g['exp07_only'] += bool(a and not b)
            g['exp12_only'] += bool(b and not a)
            for path in structure['topk_paths']:
                for method in ('exp07', 'exp12'):
                    g['states'][method][path[method]['state']] += 1
    for g in groups.values():
        g['states'] = {m: dict(v) for m, v in g['states'].items()}
        g['mean_transferred_slots'] = g['transferred_slots']/g['questions']
        g['mean_anchor_fraction'] = g.pop('anchor_fraction_sum')/g['questions']
    return groups


def freeze_panel(vrows, mrows, banks):
    """接口：12个固定槽位按源清单顺序及视频去重；空槽不跨类补足，使用归一化MLVU计分。"""
    slots, panel, used = [], [], set()
    def take(name, dataset, rows, predicate, profiles):
        q = None
        for row in rows:
            if (dataset, row['video_id']) in used or row.get('ambiguous_correct_options', False):
                continue
            if dataset == 'videomme' and row['question_id'] in ('491-2', '321-2'):
                continue
            if predicate(row['question_id']):
                q = row['question_id']
                used.add((dataset, row['video_id']))
                panel.append(dict(dataset=dataset, question_id=q, slot=name, profiles=profiles,
                                  video_id=row['video_id']))
                break
        slots.append(dict(slot=name, dataset=dataset, question_id=q))
    mb = banks['mlvu/qwen32']
    def cell(q):
        return tuple(mb[q]['answers'][m]['correct'] for m in (METHODS[1], METHODS[2], METHODS[3]))
    for i in range(2):
        take('needle_topk_only_'+str(i+1), 'mlvu', [r for r in mrows if r['task_type']=='needle'],
             lambda q: cell(q)==(True, False, False), ['qwen32', 'llava8'])
    take('needle_rd_gain', 'mlvu', [r for r in mrows if r['task_type']=='needle'],
         lambda q: not cell(q)[0] and any(cell(q)[1:]), ['qwen32', 'llava8'])
    take('needle_protection', 'mlvu', [r for r in mrows if r['task_type']=='needle'],
         lambda q: all(cell(q)) and mb[q]['structure']['selected_overlap'] < mb[q]['structure']['actual_budget'], ['qwen32', 'llava8'])
    for a in (True, False):
        take('plot_exp07_only' if a else 'plot_exp12_only', 'mlvu', [r for r in mrows if r['task_type']=='plotQA'],
             lambda q, a=a: cell(q)[1:] == (a, not a), ['qwen32', 'llava8'])
    for task in ('anomaly_reco', 'topic_reasoning'):
        take(task+'_rd_gain', 'mlvu', [r for r in mrows if r['task_type']==task],
             lambda q: cell(q)==(False, True, True), ['qwen32', 'llava8'])
    def vc(profile, q):
        item = banks['videomme/'+profile][q]
        return tuple(item['answers'][m]['correct'] for m in (METHODS[2], METHODS[3]))
    take('videomme_llava8_exp07_only', 'videomme', vrows,
         lambda q: vc('llava8', q)==(True, False), ['llava8', 'qwen32'])
    take('videomme_qwen32_exp12_only', 'videomme', vrows,
         lambda q: vc('qwen32', q)==(False, True), ['qwen32', 'llava8'])
    take('videomme_qwen16_to32_reversal', 'videomme', vrows,
         lambda q: vc('qwen16', q)==(True, False) and vc('qwen32', q)==(False, True), ['qwen32', 'llava8'])
    take('videomme_llava8_to16_reversal', 'videomme', vrows,
         lambda q: vc('llava8', q)==(True, False) and vc('llava16', q)==(False, True), ['llava8', 'qwen32'])
    return dict(questions=panel, slots=slots, maximum_questions=12, visual_minutes_limit=60,
                rule='frozen source order, unique content videos; empty slots remain empty',
                posthoc_review=True, user_agreement_rate=None)


def analyze(root):
    """接口：冻结小测清单后分析9748个主工作点；零生成、零模型编码，支持完整恢复核验。"""
    root = Path(root)
    if (root/'summary.json').exists():
        return verify(root)
    reader = Reader()
    for name in FILES:
        reader.bind(ROOT/name)
    vrows = reader.read(ROOT/'data/manifests/video_mme_full_2700.json')['rows']
    mrows = reader.read(ML/'manifest.json')['rows']
    if len(vrows)!=2700 or len(mrows)!=2174:
        raise ValueError('Main denominator differs')
    # 步骤1：只投影三个身份字段，必须先于读取任何答案冻结MLVU210题。
    subset = freeze_balanced_subset([{k:r[k] for k in ('question_id','task_type','video_id')} for r in mrows])
    publish(root/'mlvu_development_210.json', subset)
    dev = reader.read(ROOT/'data/manifests/video_mme_development_200_20260920.json')
    publish(root/'future_test_inputs.json', dict(videomme_question_ids=[r['question_id'] for r in dev['rows']],
        mlvu_question_ids=[r['question_id'] for r in subset['rows']], profiles=['llava8','qwen32'],
        future_qa_ceiling=820, generation_authorized=False, formula_review_required=True))
    publish(root/'protocol.json', dict(code={p:reader.sources[str(ROOT/p)] for p in FILES},
        main_questions={'videomme':2700,'mlvu':2174}, profiles=['llava8','qwen32'],
        policy=PERIOD_POLICY, repetitions=10000, seed=2027, new_model_calls=0,
        source_runs=[str(p) for p in (ML, BASE, RUN8, RUN32, Q12, RUN16, POOL16, S2)],
        frozen_subset_sha256=sha256(root/'mlvu_development_210.json'),
        candidate_implementation_authorized=False, visual_review_minutes_limit=60))
    records = {}; reports = {}; structure_reports = {}; witness_reports = {}
    facts = reader.read(FACTS)['entries']
    if len(facts)!=75 or len({f['question_id'] for f in facts})!=42:
        raise ValueError('Frozen witness denominator differs')
    started = time.perf_counter()
    # 步骤2：完整配对，独立核验实际输入与共同前缀，并保存被替换Top-K的可行路径。
    for dataset, data in [('videomme',vrows), ('mlvu',mrows)]:
        for profile in ('llava8','qwen32'):
            label = dataset+'/'+profile
            entries = {}; fact_paths = []
            for pos, row in enumerate(data, 1):
                q = row['question_id']
                pool, a, b, pp, fp = pool_pair(dataset, profile, q, reader)
                structure = audit_selection_paths(pool, a, b, 8 if profile=='llava8' else 32)
                answers, sources, inputs = {}, {}, {}
                for method in METHODS:
                    p = result_path(dataset, profile, method, q)
                    item = reader.read(p)
                    answers[method] = answer_record(row, item, dataset)
                    sources[method] = str(p)
                    frames = item['selection']['selected_frames']
                    if len(frames) != len(item['rgb_sha256']):
                        raise ValueError('RGB count differs')
                    identities = [frame_identity(f) for f in frames]
                    if len(set(identities)) != len(identities) or identities != sorted(identities, key=lambda x:x[1]):
                        raise ValueError('Source identity/order differs')
                    inputs[method] = dict(source_frames=identities, rgb_sha256=item['rgb_sha256'])
                    if method in METHODS[2:]:
                        indices = structure['selected_indices']['exp07' if method==METHODS[2] else 'exp12']
                        if set(identities) != {frame_identity(pool['candidates'][i]) for i in indices}:
                            raise ValueError('Saved model source differs from selection trace')
                if inputs[METHODS[2]]['source_frames'] == inputs[METHODS[3]]['source_frames'] and inputs[METHODS[2]]['rgb_sha256'] != inputs[METHODS[3]]['rgb_sha256']:
                    raise ValueError('Same source but different RGB')
                entry = dict(question_id=q, video_id=row['video_id'], task_type=row['task_type'],
                    stratum=row.get('stratum','not_applicable'), answers=answers, answer_sources=sources,
                    inputs=inputs, structure=structure, pool_source=str(pp), feature_source=str(fp))
                publish(root/'records'/dataset/profile/(q+'.json'), entry)
                entries[q] = entry
                if dataset=='videomme':
                    bypts = {c['source_pts']:frame_identity(c) for c in pool['candidates']}
                    for fact in [f for f in facts if f['question_id']==q]:
                        keys = [bypts[p] for p in fact['witness_pts'] if p in bypts]
                        fact_paths.append(dict(fact=fact, exp07=trace_witness_path(pool,a,keys),
                                               exp12=trace_witness_path(pool,b,keys)))
                if pos%100 == 0:
                    durable(root/'status.json', dict(stage='structure', dataset=dataset, profile=profile,
                            completed=pos, total=len(data), new_model_calls=0, utc=utc()))
            records[label] = entries
            policies = {}
            for policy in ('native','period_normalized') if dataset=='mlvu' else ('native',):
                bank = {m:{q:dict(correct=e['answers'][m]['native_correct'] if policy=='native' else e['answers'][m]['correct'])
                           for q,e in entries.items()} for m in METHODS}
                policies[policy] = cluster_report(data, bank, macro=dataset=='mlvu')
            reports[label] = policies
            structure_reports[label] = summarize_structure(list(entries.values()))
            if dataset=='videomme':
                witness_reports[profile] = dict(questions=42, facts=75, entries=fact_paths,
                    exact_hits={m:sum(f[m]['exact_hit'] for f in fact_paths) for m in ('exp07','exp12')},
                    limitation='Unlisted visual alternatives not automatically judged absent')
    # 步骤3：16帧只核验两版答案与来源，作为预算趋势参照，不新增选择或问答。
    for profile in ('llava16','qwen16'):
        entries = {}
        for row in vrows:
            q = row['question_id']; answers = {}; sources = {}
            for method in METHODS[2:]:
                p = result_path('videomme',profile,method,q)
                item = reader.read(p)
                answers[method] = answer_record(row,item,'videomme')
                sources[method] = str(p)
            entries[q] = dict(answers=answers, answer_sources=sources)
        records['videomme/'+profile] = entries
        reports['videomme/'+profile] = dict(native=cluster_report(vrows,
            {m:{q:dict(correct=x['answers'][m]['correct']) for q,x in entries.items()} for m in METHODS[2:]}))
    # 步骤4：BLIP2仅作为既有输入扰动参照，保留逐题方向而不重新评分。
    perturbations = {}
    for profile in ('llava8','qwen32'):
        values = {m:Counter() for m in METHODS[1:]}
        for row in vrows:
            q=row['question_id']
            for method in METHODS[1:]:
                p=S2/'qa'/profile/'results'/method/(q+'.json')
                item=reader.read(p)
                new=answer_record(row,item,'videomme')['correct']
                old=records['videomme/'+profile][q]['answers'][method]['correct']
                values[method]['improved' if new and not old else 'regressed' if old and not new else 'unchanged']+=1
        perturbations[profile]={k:dict(v) for k,v in values.items()}
    panel=freeze_panel(vrows,mrows,records)
    publish(root/'panel.json',panel)
    publish(root/'statistics.json',reports)
    publish(root/'structure_summary.json',structure_reports)
    publish(root/'witness_paths.json',witness_reports)
    publish(root/'scorer_perturbation.json',perturbations)
    publish(root/'sources.json',reader.sources)
    exports={str(p.relative_to(root)):sha256(p) for p in root.rglob('*.json') if p.name not in ('status.json','summary.json')}
    summary=dict(status='analysis_complete_review_pending', main_question_configurations=9748,
        supplemental_16frame_question_configurations=5400, panel_questions=len(panel['questions']),
        source_files=len(reader.sources), sources_sha256=sha256(root/'sources.json'), exports=exports,
        new_model_calls=0, new_answers=0, candidate_implemented=False,
        seconds=time.perf_counter()-started, utc=utc())
    durable(root/'summary.json',summary)
    return summary


def verify(root):
    """接口：完成恢复只核查字节，不新建评分、特征、选择或生成；异常立即停止。"""
    root=Path(root);summary=read_json(root/'summary.json');sources=read_json(root/'sources.json')
    if sha256(root/'sources.json')!=summary['sources_sha256']:
        raise ValueError('Source ledger changed')
    for p,h in sources.items():
        if sha256(p)!=h:
            raise ValueError('Historical source changed: '+p)
    for p,h in summary['exports'].items():
        if sha256(root/p)!=h:
            raise ValueError('Analysis export changed: '+p)
    return dict(status='passed', historical_sources=len(sources), exports=len(summary['exports']),
                new_model_calls=0,new_answers=0,completed_resume_new_analysis=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--phase',choices=('analyze','verify'),default='analyze')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):
        raise ValueError('Run ID must be a single directory name')
    destination=ROOT/'outputs/analysis'/args.run_id
    with exclusive(destination/'analysis.lock'):
        print(json.dumps(analyze(destination) if args.phase=='analyze' else verify(destination),ensure_ascii=False))
