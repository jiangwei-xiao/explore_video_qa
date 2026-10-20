"""RD成果的只读汇总：复用冻结统计、明确计分与比较资格，不调用模型或选择器。"""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, exclusive, utc

METHODS = ('BASE-Uniform', 'BASE-BLIP-TopK', 'RD-1.2-exp07', 'RD-1.2-exp12')
MODELS = {'llava8': ('LLaVA-Video-7B-Qwen2', 8), 'llava16': ('LLaVA-Video-7B-Qwen2', 16),
          'qwen16': ('Qwen2.5-VL-7B-Instruct', 16), 'qwen32': ('Qwen2.5-VL-7B-Instruct', 32)}
REVIEW = 'outputs/analysis/dual_dataset_selection_review_20261015_r2'
FULL = 'outputs/full_videoqa/'
RUNS = dict(llava8='llava8__five_methods__videomme2700__20261005_r1',
            qwen32='uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01',
            exp07='rd12_exp07__qwen32__videomme2700__20260930_r1',
            exp12='rd12_exp12__qwen32__videomme2700__20261003_r1',
            exp07_16='exp07_budget16__videomme2700__20261005_r1',
            gen='exp12_generalization__videomme2700__20261005_r1',
            llava16='topk_rd12__llava__lmms_ref_v1__videomme2700__20260923__r01',
            qwen16='uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__20260923__r01',
            mlvu='mlvu_dev__S1__llava8_qwen32__four_methods__20261014_r1',
            scorer='scorer_comparison__blip2__qwen32_llava8__20261014_r1')
CATALOG = 'docs/experiments/cross_dataset/2026-10-20/evidence_consolidation/evidence_catalog.json'
MLVU_TASKS = {'plotQA', 'needle', 'ego', 'count', 'order', 'anomaly_reco', 'topic_reasoning'}


class EvidenceReader:
    """接口：记录实际引用文件的字节摘要；恢复拒绝来源变化，不重扫视频及特征。"""
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.sources = {}
        self.documents = {}

    def path(self, relative):
        """接口：只接受仓库内来源，防止错误目录被当作冻结结果。"""
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Evidence source outside repository')
        return path

    def check(self, relative, expected=None):
        """步骤1检查既有完成指纹；步骤2保存本轮实际读取来源的摘要。"""
        path = self.path(relative); key = str(path.relative_to(self.root)); value = sha256(path)
        if expected is not None and value != expected:
            raise ValueError('Frozen evidence fingerprint differs: ' + key)
        if key in self.sources and value != self.sources[key]:
            raise ValueError('Evidence changed during collection: ' + key)
        self.sources[key] = value
        return path

    def read(self, relative, expected=None):
        """接口：读取JSON元数据，不递归打开其中的模型、视频或原始结果路径。"""
        path = self.check(relative, expected)
        key = str(path.relative_to(self.root))
        if key not in self.documents:
            self.documents[key] = read_json(path)
        return self.documents[key]

    def verify(self):
        """接口：只核验实际引用文件；不执行原实验入口或全仓保护扫描。"""
        for relative, expected in self.sources.items():
            if sha256(self.path(relative)) != expected:
                raise ValueError('Evidence source mutated: ' + relative)


def pointer(*parts):
    """接口：构造JSON Pointer，正确保留含斜线的配置键与方法名。"""
    return '/' + '/'.join(str(p).replace('~', '~0').replace('/', '~1') for p in parts)


def resolve_pointer(document, location):
    """接口：独立定位结果来源，避免存在哈希但指向错误字段的展示记录。"""
    if not location.startswith('/'):
        raise ValueError('Expected absolute JSON pointer')
    for part in location[1:].split('/'):
        key = part.replace('~1', '/').replace('~0', '~')
        document = document[int(key)] if isinstance(document, list) else document[key]
    return document


def metric_record(raw, metric, expected_n=None):
    """接口：统一总体与宏平均字段，禁止把MLVU微平均当成M-Avg。"""
    # 步骤1校验分母和正确数；保留原始精度，不先四舍五入再计算增益。
    item = raw.get('all', raw)
    n, correct = int(item['n']), int(item['correct'])
    if n <= 0 or not 0 <= correct <= n or (expected_n is not None and n != expected_n):
        raise ValueError('Metric population differs')
    accuracy = 100 * correct / n
    if 'accuracy' in item and abs(100 * item['accuracy'] - accuracy) > 1e-8:
        raise ValueError('Accuracy/count mismatch')
    if 'accuracy_percent' in item and abs(item['accuracy_percent'] - accuracy) > 1e-8:
        raise ValueError('Accuracy/count mismatch')
    score = accuracy
    tasks = item.get('tasks', raw.get('tasks'))
    # 步骤2 MLVU必须保留七个任务分母；用既有任务正确数核对现成宏平均，不重新生成答案。
    if metric == 'M-Avg':
        if not tasks or set(tasks) != MLVU_TASKS:
            raise ValueError('MLVU requires exactly seven task denominators')
        if sum(t['n'] for t in tasks.values()) != n or sum(t['correct'] for t in tasks.values()) != correct:
            raise ValueError('MLVU task population/count mismatch')
        score = 100 * sum(t['correct'] / t['n'] for t in tasks.values()) / 7
        existing = item.get('main_metric_percent', item.get('m_avg_percent'))
        if existing is None or abs(score - existing) > 1e-8:
            raise ValueError('MLVU macro/micro score mismatch')
    elif metric != 'accuracy':
        raise ValueError('Unknown metric')
    return dict(n=n, correct=correct, score_percent=score, accuracy_percent=accuracy,
                metric=metric, wilson_95=item.get('wilson_95'), tasks=tasks)


def paired_record(left, right, raw, population=None, reverse=False):
    """接口：复用既有配对区间；反向比较同时交换得失并反转区间端点。"""
    # 步骤1核对四格完整互斥；精度与M-Avg差值仍按来源的原指标解释。
    cells = raw['cells']; required = {'both_correct', 'left_only', 'right_only', 'both_wrong'}
    if set(cells) != required:
        raise ValueError('Paired four-grid schema differs')
    flat = [q for k in required for q in cells[k]]
    if len(set(flat)) != len(flat) or (population is not None and len(flat) != population):
        raise ValueError('Paired question population differs')
    ci = raw.get('video_cluster_95_pp', raw.get('stratified_video_cluster_bootstrap_95'))
    if ci is None or len(ci) != 2 or not all(math.isfinite(x) for x in ci) or ci[0] > ci[1]:
        raise ValueError('Missing/invalid paired interval')
    delta = float(raw['difference_pp'])
    if reverse:
        left, right = right, left
        delta, ci = -delta, [-ci[1], -ci[0]]
        gains, losses = cells['right_only'], cells['left_only']
    else:
        gains, losses = cells['left_only'], cells['right_only']
    # 步骤2只保存完整题单的来源指针，避免复制大量重复题号或重新bootstrap。
    return dict(left=left, right=right, n=len(flat), difference_pp=delta,
                interval_95_pp=list(ci), improved=len(gains), regressed=len(losses),
                net_correct=len(gains) - len(losses), both_correct=len(cells['both_correct']),
                both_wrong=len(cells['both_wrong']), source_reversed=reverse)


def comparison_qualification(local, external):
    """接口：判断比较资格；作者报告不视为本机配对，归一化计分不直接比较作者分数。"""
    if external.get('score') is None:
        return 'not_reported_for_target_configuration'
    if local['profile'] != external['profile'] or local['dataset'] != external['dataset']:
        return 'different_model_budget_or_dataset'
    if local['score_version'] != 'native':
        return 'different_scoring_version_context_only'
    if external['budget_policy'] != 'fixed':
        return 'same_budget_ceiling_auxiliary_analysis_context_only'
    return 'same_named_setting_author_reported_not_local_paired'


def collect_tables(reader):
    """接口：冻结汇总与既有配对→主表/补充表/子集；不读取原片或重放选择。"""
    # 步骤1优先复用已审计的跨数据集统计，并核验其导出哈希。
    previous = reader.read(REVIEW + '/summary.json')
    stats_path = REVIEW + '/statistics.json'
    stats = reader.read(stats_path, previous['exports']['statistics.json'])
    reader.check(REVIEW + '/sources.json', previous['sources_sha256'])
    registry = reader.read('configs/method_versions.json')
    if not set((*METHODS, 'RD-1.2')).issubset({r['id'] for r in registry['entries']}):
        raise ValueError('Canonical method registration missing')
    expected_keys = {'videomme/llava8', 'videomme/qwen32', 'mlvu/llava8', 'mlvu/qwen32',
                     'videomme/llava16', 'videomme/qwen16'}
    if set(stats) != expected_keys:
        raise ValueError('Frozen full-dataset configuration population differs')
    snapshots = {name: reader.read(FULL + run + '/summary.json') for name, run in RUNS.items()
                 if name not in ('scorer', 'mlvu')}
    sc_path = FULL + RUNS['scorer'] + '/comparison.json'
    sc_accept = reader.read(FULL + RUNS['scorer'] + '/completion_acceptance.json')
    scorer = reader.read(sc_path, sc_accept['source_artifacts']['comparison.json'])
    ml_path = FULL + RUNS['mlvu'] + '/comparison.json'
    ml_accept = reader.read(FULL + RUNS['mlvu'] + '/completion_acceptance.json')
    mlvu = reader.read(ml_path, ml_accept['comparison_sha256'])
    ml_frames_path = FULL + RUNS['mlvu'] + '/independent_selection_complete.json'
    ml_frames = reader.read(ml_frames_path)
    l8_input = reader.read(FULL + RUNS['llava8'] + '/input_summary.json')
    q32_frames = snapshots['exp12']['frame_counts']
    base16 = {p: snapshots[p] for p in ('llava16', 'qwen16')}
    rows, pairs, subsets, missing = [], [], [], []

    def add_row(dataset, profile, version, method, value, path, pointer,
                frame_counts=None, section='main', frontend='S1', sharing=None, protocol_path=None):
        """接口：每行绑定模型、K、计分、来源及原协议哈希；缺测不能填0。"""
        n = 2700 if dataset == 'videomme' else 2174
        metric = 'accuracy' if dataset == 'videomme' else 'M-Avg'
        score = metric_record(value, metric, n)
        model, budget = MODELS[profile]
        item = dict(dataset=dataset, data_scope='2700 questions/900 videos/no subtitles' if dataset == 'videomme'
                    else 'MLVU-Dev seven MCQ tasks/2174 questions', profile=profile, model=model,
                    frame_budget=budget, method=method, section=section, score_version=version,
                    comparison_frontend=frontend, scorer_id=None if method == 'BASE-Uniform' else frontend,
                    source_path=path, source_sha256=reader.sources[path], source_pointer=pointer,
                    actual_frame_counts=frame_counts, sharing=sharing, **score)
        if frame_counts is not None and (sum(frame_counts.values()) != n or
                any(not 0 < int(k) <= budget for k in frame_counts)):
            raise ValueError('Actual frame histogram violates source budget/population')
        if protocol_path is not None:
            protocol = reader.read(protocol_path)
            model_value = protocol.get('model')
            if isinstance(model_value, str) and model_value != model:
                raise ValueError('Model identity differs from profile')
            profile_value = protocol.get('profile')
            declared = protocol.get('source_frame_budget', protocol.get('frame_budget'))
            if declared is None and isinstance(profile_value, dict):
                declared = profile_value.get('frame_budget')
            if declared is not None and declared != budget:
                raise ValueError('Protocol frame budget differs')
            item.update(protocol_path=protocol_path, protocol_sha256=reader.sources[protocol_path])
        item['task_protocol'] = ('lmms-videomme-reference-v1' if dataset == 'videomme' else
                                 'fixed-lmms-mlvu-task/native-plus-separate-period-supplement')
        rows.append(item)
        return item

    def add_subsets(item, values, source_path, pointer, metric='accuracy'):
        """接口：保留原批次子集定义；不把S1/S2共同满预算偷换为仅S1共同子集。"""
        for name, value in values.items():
            score = metric_record(value, metric)
            subsets.append(dict(dataset=item['dataset'], profile=item['profile'], method=item['method'],
                                score_version=item['score_version'], comparison_frontend=item['comparison_frontend'],
                                subset=name, source_path=source_path, source_sha256=reader.sources[source_path],
                                source_pointer=pointer + '/' + name.replace('~', '~0').replace('/', '~1'),
                                common_budget_scope='source-run-defined; retain original compared methods', **score))

    for key, variants in stats.items():
        dataset, profile = key.split('/')
        if set(variants) != ({'native', 'period_normalized'} if dataset == 'mlvu' else {'native'}):
            raise ValueError('Score versions differ from frozen review')
        for version, report in variants.items():
            if dataset == 'mlvu' and profile == 'qwen32' and version == 'period_normalized':
                # Qwen无标点修复，避免重复展示相同工作点为额外证据。
                if report['scores'] != variants['native']['scores']:
                    raise ValueError('Qwen normalization unexpectedly changes scores')
                continue
            expected_methods = set(METHODS[2:]) if profile in ('llava16', 'qwen16') else set(METHODS)
            if set(report['scores']) != expected_methods:
                raise ValueError('Method identity/population differs')
            for method, score in report['scores'].items():
                is16 = profile in ('llava16', 'qwen16')
                if dataset == 'mlvu':
                    frames = ml_frames['profiles'][profile]['frame_counts'][method]
                    proto = FULL + RUNS['mlvu'] + '/qa/' + profile + '/protocol.json'
                    sharing = mlvu['profiles'][profile]['input_summary']
                elif not is16:
                    if profile == 'llava8' and l8_input['frame_counts'][method] != [8]:
                        raise ValueError('LLaVA8 actual source budget changed')
                    frames = ({'8': 2700} if profile == 'llava8' else q32_frames[method])
                    origin = ('llava8' if profile == 'llava8' else 'exp07' if method.endswith('exp07')
                              else 'exp12' if method.endswith('exp12') else 'qwen32')
                    proto = FULL + RUNS[origin] + '/protocol.json'
                    sharing = ('11932 unique calls/1568 same-input method references across five methods'
                               if profile == 'llava8' else 'exp07/exp12 reused 200 development answers; see individual source runs')
                else:
                    frames = {'16': 2700}
                    proto = FULL + RUNS['exp07_16' if method.endswith('exp07') else 'gen'] + '/' + profile + '/protocol.json'
                    sharing = 'exp07:797 same-input imports+1903 new calls; exp12:2700 new calls per model'
                item = add_row(dataset, profile, version, method, score, stats_path,
                               pointer(key, version, 'scores', method), frames,
                               'supplement16' if is16 else 'main', sharing=sharing, protocol_path=proto)
                if dataset == 'videomme' and not is16:
                    values = scorer['profiles'][profile]['scores']['S1'][method]
                    add_subsets(item, values, sc_path, pointer('profiles', profile, 'scores', 'S1', method))
                    if abs(metric_record(values['all2700'], 'accuracy')['score_percent'] - item['score_percent']) > 1e-8:
                        raise ValueError('S1 source summaries disagree')
                elif is16:
                    parent = snapshots['exp07_16']['reports'][profile] if method.endswith('exp07') else snapshots['gen']['reports'][profile]
                    pth = FULL + RUNS['exp07_16' if method.endswith('exp07') else 'gen'] + '/summary.json'
                    add_subsets(item, parent['metrics'], pth, pointer('reports', profile, 'metrics'))
                elif version == 'native':
                    add_subsets(item, mlvu['profiles'][profile]['scores'][method], ml_path,
                                pointer('profiles', profile, 'scores', method), 'M-Avg')
            for label, value in report['pairs'].items():
                left, right = label.split(' minus ')
                reverse = METHODS.index(left) < METHODS.index(right)
                record = paired_record(left, right, value, 2700 if dataset == 'videomme' else 2174, reverse)
                lookup = {m: report['scores'][m]['main_metric_percent'] for m in (record['left'], record['right'])}
                if abs(record['difference_pp'] - (lookup[record['left']] - lookup[record['right']])) > 1e-8:
                    raise ValueError('Paired metric direction/score differs')
                pairs.append(dict(dataset=dataset, profile=profile, score_version=version, comparison_frontend='S1',
                                  source_path=stats_path, source_sha256=reader.sources[stats_path],
                                  source_pointer=pointer(key, version, 'pairs', label), **record))

    # 步骤2补齐原RD及16帧旧基线，缺测MLVU原RD明确保存为缺失。
    for profile in ('llava8', 'qwen32'):
        p = FULL + RUNS['llava8' if profile == 'llava8' else 'qwen32'] + '/summary.json'
        value = (snapshots['llava8']['metrics']['RD-1.2']['all2700'] if profile == 'llava8' else
                 snapshots['qwen32']['accuracy']['K32__RD-1.2']['all2700'])
        frames = {'8': 2700} if profile == 'llava8' else q32_frames['RD-1.2']
        location = pointer('metrics', 'RD-1.2', 'all2700') if profile == 'llava8' else pointer('accuracy', 'K32__RD-1.2', 'all2700')
        add_row('videomme', profile, 'native', 'RD-1.2', value, p, location, frames,
                protocol_path=FULL + RUNS['llava8' if profile == 'llava8' else 'qwen32'] + '/protocol.json')
        missing.append(dict(dataset='mlvu', profile=profile, method='RD-1.2', score=None,
                            status='not_evaluated', reason='MLVU run included four methods; original RD not tested'))
    for profile in ('llava16', 'qwen16'):
        path = FULL + RUNS[profile] + '/summary.json'; data = base16[profile]
        for method in ('BASE-Uniform', 'BASE-BLIP-TopK', 'RD-1.2'):
            key = method if profile == 'llava16' else 'qwen__' + method
            frames = data['issues'][key].get('frame_counts', data['issues'][key].get('source_frame_counts'))
            item = add_row('videomme', profile, 'native', method, data['accuracy'][key]['all2700'], path,
                           pointer('accuracy', key, 'all2700'), frames, 'supplement16',
                           protocol_path=FULL + RUNS[profile] + '/protocol.json')
            add_subsets(item, data['accuracy'][key], path, pointer('accuracy', key))
        for method in ('RD-1.2-exp07', 'RD-1.2-exp12'):
            which = 'exp07_16' if method.endswith('exp07') else 'gen'
            pth = FULL + RUNS[which] + '/summary.json'; stage = snapshots[which]['reports'][profile]
            groups = stage['comparisons' if which == 'exp07_16' else 'paired']
            for baseline, scopes in groups.items():
                if baseline not in ('BASE-Uniform', 'BASE-BLIP-TopK', 'RD-1.2'):
                    continue
                record = paired_record(method, baseline, scopes['all2700'], 2700)
                pairs.append(dict(dataset='videomme', profile=profile, score_version='native', comparison_frontend='S1',
                                  source_path=pth, source_sha256=reader.sources[pth],
                                  source_pointer=pointer('reports', profile,
                                      'comparisons' if which == 'exp07_16' else 'paired', baseline, 'all2700'), **record))

    # 步骤3评分器扰动逐项列出，不把均匀重复计算为S2的新模型实验。
    perturbations = []
    for profile, value in scorer['profiles'].items():
        for method in METHODS[1:]:
            original = value['scores']['S1'][method]['all2700']
            changed = value['scores']['S2'][method]['all2700']
            pair = paired_record('S2/' + method, 'S1/' + method,
                                 value['paired']['S2_minus_S1__' + method]['all2700'], 2700)
            perturbations.append(dict(profile=profile, method=method, n=2700,
                                      s1=metric_record(original, 'accuracy'), s2=metric_record(changed, 'accuracy'),
                                      comparison=pair, source_path=sc_path, source_sha256=reader.sources[sc_path]))

    wfs = snapshots['gen']['reports']['wfs32']; wfs_path = FULL + RUNS['gen'] + '/summary.json'
    local_wfs = dict(metric_record(wfs['metrics']['all2700'], 'accuracy', 2700),
                     method='REF-WFS-SB', dataset='videomme', profile='qwen32', frame_budget=32,
                     kind='published_frame_list_local_qa', new_calls=2500, imported=200,
                     source_path=wfs_path, source_sha256=reader.sources[wfs_path],
                     actual_frame_counts=wfs['actual_frame_counts'],
                     paired={name: paired_record('REF-WFS-SB', name, values['all2700'], 2700)
                             for name, values in wfs['paired'].items()},
                     limitation='Same local QA backend; published source frames only, not full selection pipeline/cost reproduction')
    # 步骤4每个数值记录必须能从已引用JSON指针重建，不只检查文件存在。
    for item in rows + subsets:
        original = resolve_pointer(reader.documents[item['source_path']], item['source_pointer'])
        check = metric_record(original, item['metric'], item['n'])
        if check['correct'] != item['correct'] or abs(check['score_percent'] - item['score_percent']) > 1e-8:
            raise ValueError('Score provenance pointer differs')
    return dict(rows=rows, paired=pairs, subsets=subsets, missing=missing,
                scorer_perturbations=perturbations, local_wfs=local_wfs), snapshots, mlvu


def collect_references(reader, catalog):
    """接口：用固定本地PDF和页码锚点核对作者数字，缺少目标配置时输出NR。"""
    from pypdf import PdfReader
    inventory = reader.read('docs/related_work/pdfs/sources.json')
    metadata = {r['file']: r for r in inventory}; pages = {}; checks = []
    # 步骤1仅检查四篇已选文献及其被引用页面，PDF不下载、不写回。
    used = {r['paper'] for r in catalog['reference_rows']} | {r['paper'] for r in catalog['original_paper_gaps']}
    for paper in sorted(used):
        path = reader.check('docs/related_work/pdfs/' + paper, metadata[paper]['sha256'])
        pdf = PdfReader(path)
        requested = {r['page'] for r in catalog['reference_rows'] if r['paper'] == paper}
        requested.update(i for r in catalog['original_paper_gaps'] if r['paper'] == paper for i in r['pages'])
        for page in sorted(requested):
            text = (pdf.pages[page - 1].extract_text() or '').encode('utf8', 'replace').decode()
            pages[(paper, page)] = ''.join(text.split())
            checks.append(dict(paper=paper, page=page, pdf_sha256=metadata[paper]['sha256'],
                               text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    results = []
    for value in catalog['reference_rows']:
        if value['anchor'] not in pages[(value['paper'], value['page'])]:
            raise ValueError('Primary PDF numeric/table anchor differs: ' + str(value))
        record = dict(value, difference_pp=value['score'] - value['base'],
                      arxiv_version=metadata[value['paper']]['arxiv_version'],
                      source_url=metadata[value['paper']]['versioned_url'], pdf_sha256=metadata[value['paper']]['sha256'],
                      comparison_qualification='author_reported_context_only; environment_and_parser_not_fully_identical')
        results.append(record)
    # 步骤2原作者缺测独立列出；不能用转引复测填充“原文结果”。
    for gap in catalog['original_paper_gaps']:
        for profile in gap['profiles']:
            for dataset in gap['datasets']:
                results.append(dict(method=gap['method'], reporter='original authors', paper=gap['paper'],
                                    pages=gap['pages'], profile=profile, dataset=dataset, base=None, score=None,
                                    difference_pp=None, status='not_reported_for_target_configuration',
                                    reason=gap['reason'], pdf_sha256=metadata[gap['paper']]['sha256']))
    return results, checks


def collect_costs(reader, snapshots):
    """接口：保留实测阶段边界，整批墙钟与累计工作秒不相加冒充单题E2E。"""
    path = FULL + RUNS['mlvu'] + '/execution_costs.json'; ml = reader.read(path)
    totals = ml['preparation_worker_totals']; costs = []
    for field in ('preprocess_seconds', 'forward_seconds'):
        costs.append(dict(dataset='mlvu', component='BLIP-1 coarse ' + field,
                          seconds=totals['coarse'][field], basis='worker-summed shared phase over 2174 questions',
                          itm_frames=totals['coarse']['itm_frames'], source=path,
                          cost_class='measured_shared_phase', not_independent_e2e=True))
    for profile in ('llava8', 'qwen32'):
        for field in ('fine_decode_seconds', 'fine_visual_preprocess_seconds', 'fine_visual_forward_seconds'):
            costs.append(dict(dataset='mlvu', profile=profile, component=field, seconds=totals[profile][field],
                              visual_frames=totals[profile]['fine_visual_frames'], fine_itm_frames=totals[profile]['fine_itm_frames'],
                              basis='worker-summed shared candidate augmentation', source=path,
                              cost_class='measured_shared_phase', not_independent_e2e=True))
    costs.append(dict(dataset='mlvu', component='pipeline wall', seconds=ml['original_pipeline_wall_seconds'],
                      basis='whole multi-stage shared experiment', source=path,
                      cost_class='measured_batch_wall', not_independent_e2e=True))
    parent = snapshots['exp12']; path = FULL + RUNS['exp12'] + '/summary.json'
    costs.append(dict(dataset='videomme', profile='qwen32', method='RD-1.2-exp12', component='QA',
                      measurement=parent['performance']['qa_seconds'], source=path,
                      basis='new 2500 questions only; reused 200 costs excluded',
                      cost_class='measured_cached_input_qa', not_independent_e2e=True))
    costs.append(dict(dataset='videomme', profile='qwen32', method='RD-1.2-exp12', component='final_decode_seconds',
                      measurement=parent['performance']['final_decode_seconds'], source=path,
                      basis='original cached-input decode field; preserve original timing boundary',
                      cost_class='measured_cached_input_decode', not_independent_e2e=True))
    timing_path = FULL + RUNS['mlvu'] + '/selection_timing/acceptance.json'
    timing = reader.read(timing_path)
    if timing['status'] != 'passed' or timing['new_model_calls'] != 0:
        raise ValueError('Cached CPU selection timing acceptance differs')
    for profile, record in timing['profiles'].items():
        for method, measurement in record['methods'].items():
            costs.append(dict(dataset='mlvu', profile=profile, method=method,
                              component='CPU selection with existing candidates/features', measurement=measurement,
                              basis='previous accepted cache replay; candidate preparation excluded', source=timing_path,
                              cost_class='measured_cached_cpu_selection', not_independent_e2e=True))
    costs.append(dict(component='independent raw-video E2E comparable across methods', seconds=None,
                      cost_class='not_measured', reason='No matched standalone RD/WFS runs with consistent boundaries'))
    return costs


def render_tables(output, evidence):
    """接口：导出保持原始精度的CSV和可读Markdown；缺测写NR，不填0或拼最优方法。"""
    output = Path(output)
    # 步骤1结构化结果保留元数据/来源指针，CSV仅选择便于核对的标量字段。
    fields = ['dataset', 'profile', 'method', 'section', 'score_version', 'comparison_frontend', 'scorer_id',
              'metric', 'n', 'correct', 'score_percent', 'accuracy_percent', 'frame_budget', 'model',
              'source_path', 'source_sha256', 'source_pointer']
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(evidence['rows'])
    lines = ['# RD冻结全量证据总表', '',
             'Video-MME为准确率；MLVU为七类M-Avg。原生和标点归一化不混算，作者数据另列。', '',
             '| 数据集/配置/计分 | 方法 | 正确数 | 主指标 | Δ均匀 pp |', '|---|---|---:|---:|---:|']
    for row in evidence['rows']:
        base = next(r for r in evidence['rows'] if r['dataset'] == row['dataset'] and r['profile'] == row['profile']
                    and r['score_version'] == row['score_version'] and r['method'] == 'BASE-Uniform')
        lines.append(f"| {row['dataset']}/{row['profile']}/{row['score_version']} | {row['method']} | "
                     f"{row['correct']}/{row['n']} | {row['score_percent']:.2f}% | {row['score_percent']-base['score_percent']:+.3f} |")
    lines += ['', 'MLVU原RD-1.2未测，不补0；实际帧数、分组和来源见evidence.json、subsets.json。', '',
              '## BLIP-2扰动', '', '| 配置 | 方法 | S1正确 | S2正确 | S2−S1 pp | 95%区间 pp |',
              '|---|---|---:|---:|---:|---|']
    for p in evidence['scorer_perturbations']:
        pair = p['comparison']; lo, hi = pair['interval_95_pp']
        lines.append(f"| {p['profile']} | {p['method']} | {p['s1']['correct']} | {p['s2']['correct']} | "
                     f"{pair['difference_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', '## 相关工作：作者报告', '',
              '| 数据集/配置 | 方法 | 报告者 | 原均匀→方法 | Δ pp | 位置 |', '|---|---|---|---|---:|---|']
    for r in evidence['references']:
        if r['score'] is None:
            values, delta = 'NR', '—'
        else:
            values, delta = f"{r['base']:.1f}→{r['score']:.1f}", f"{r['difference_pp']:+.1f}"
        page = r.get('page', ','.join(map(str, r.get('pages', []))))
        lines.append(f"| {r['dataset']}/{r['profile']} | {r['method']} | {r['reporter']} | "
                     f"{values} | {delta} | {r['paper']} p{page} |")
    wfs = evidence['local_wfs']
    lines += ['', f"本机REF-WFS-SB发布选帧Qwen32为{wfs['correct']}/2700（{wfs['score_percent']:.3f}%）；"
              '它是独立的同QA后端对照，不包含完整WFS评分/选择成本。', '',
              '## 成本边界', '', '粗ITM与细帧视觉计算见costs.json。共享阶段工作秒和批次墙钟不能相加作为独立E2E。',
              '计数模型：T_retrieval = Nc·(C_visual + C_ITM(question)) + Nfine·C_visual；'
              '固定前缀DP的转移上界O(RK²)，局部前缀涉及特征比较与追踪开销。它们不是实测速度或答案充分性保证。',
              '', '原生作者报告与本机标点归一化结果不作直接排序；跨论文数据没有本机配对显著性。', '']
    (output / 'tables.md').write_text('\n'.join(lines))


def run(output, root=ROOT):
    """接口：只读冻结来源，写独立分析目录；完成恢复只验指纹，不重算或改写旧统计。"""
    output = Path(output); output.mkdir(parents=True, exist_ok=True); reader = EvidenceReader(root)
    with exclusive(output / 'analysis.lock'):
        if (output / 'completion.json').exists():
            saved = read_json(output / 'completion.json')
            for p, expected in saved['sources'].items():
                if sha256(reader.path(p)) != expected:
                    raise ValueError('Completed evidence source changed: ' + p)
            for p, expected in saved['artifacts'].items():
                if sha256(output / p) != expected:
                    raise ValueError('Completed evidence output changed: ' + p)
            return read_json(output / 'evidence.json')
        began = time.monotonic()
        # 步骤1绑定本轮实现和人工核对目录；不访问模型权重或调用原实验协调器。
        for name in ('src/videoqa_audit/rd_evidence.py', 'scripts/summarize_rd_evidence.py'):
            reader.check(name)
        catalog = reader.read(CATALOG)
        tables, snapshots, mlvu = collect_tables(reader)
        references, checks = collect_references(reader, catalog)
        reference_comparisons = []
        for local in tables['rows']:
            if local['section'] != 'main' or local['method'] not in METHODS[2:]:
                continue
            for reference in references:
                if reference['profile'] != local['profile'] or reference['dataset'] != local['dataset']:
                    continue
                qualification = comparison_qualification(local, reference)
                reference_comparisons.append(dict(dataset=local['dataset'], profile=local['profile'],
                    method=local['method'], score_version=local['score_version'],
                    external_method=reference['method'], reporter=reference['reporter'],
                    qualification=qualification, paired=False,
                    contextual_absolute_gap_pp=(local['score_percent']-reference['score']
                        if qualification=='same_named_setting_author_reported_not_local_paired' else None)))
        costs = collect_costs(reader, snapshots)
        for claim in catalog['claim_matrix']:
            for name in claim['evidence']:
                reader.check(name)
        for name in ('src/videoqa_methods/prefix_coverage_budget.py', 'src/videoqa_methods/relevance_coverage_budget.py',
                     'src/videoqa_methods/coarse_eligibility.py', 'src/videoqa_methods/weighted_seed_support.py'):
            reader.check(name)
        # 步骤2把近期停止结论也绑定为研究来源，不能把诊断分数塞入全量方法表。
        cpu_path = 'outputs/analysis/longvideo_pilot_trainability__197cv194__20261020_r1/report.json'
        cpu = reader.read(cpu_path)
        if cpu['recommendation']['passed'] or cpu['primary']['correct']['learned_oof'] != 129:
            raise ValueError('Current set-head stop evidence differs')
        evidence = dict(**tables, references=references, reference_comparisons=reference_comparisons,
                        primary_pdf_checks=checks, costs=costs,
                        claims=catalog['claim_matrix'], candidate_contributions=catalog['candidate_contributions'],
                        recommendation=catalog['main_version_recommendation'], followups=catalog['followups'],
                        stage_decision=catalog['stage_decision'], status='completed',
                        recent_set_head=dict(source=cpu_path, correct_197=129, correct_194=127,
                                             continuing=False), new_model_calls=0, new_answers=0,
                        automatic_experiments_authorized=False, seconds=time.monotonic()-began, utc=utc())
        ids = [(r['dataset'], r['profile'], r['score_version'], r['comparison_frontend'], r['method']) for r in tables['rows']]
        if len(set(ids)) != len(ids) or len(catalog['followups']) > 2 or len(catalog['candidate_contributions']) > 3:
            raise ValueError('Duplicate rows or evidence scope exceeded')
        # 步骤3只保存派生统计和表格；所有原结果、旧计分和指纹文件保持不变。
        durable(output / 'evidence.json', evidence)
        durable(output / 'subsets.json', tables['subsets'])
        durable(output / 'paired.json', tables['paired'])
        durable(output / 'costs.json', costs)
        durable(output / 'sources.json', reader.sources)
        render_tables(output, evidence)
        reader.verify()
        artifacts = {p.name: sha256(p) for p in output.iterdir() if p.is_file()
                     and p.name not in ('completion.json', 'analysis.lock', 'coordinator.log')}
        durable(output / 'completion.json', dict(status='completed', sources=reader.sources, artifacts=artifacts,
                                                 new_model_calls=0, new_answers=0, utc=utc()))
        return evidence
