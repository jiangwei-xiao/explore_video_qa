"""exp34零模型离线重放：冻结exp33新增帧，只重选删除对象。"""
from collections import Counter
import copy
from fractions import Fraction
from pathlib import Path
import shutil
import time

import numpy as np

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, exclusive, utc
from videoqa_budget32.frames import validate_frames
from videoqa_methods.adjacent_context_repair import source_key
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.density_selection import source_time
from videoqa_methods.low_loss_replacement import select_low_loss_replacement
from videoqa_audit import adjacent_context_development as PREVIOUS
from videoqa_audit.adjacent_context_experiment import FACTS

METHOD = 'RD-1.2-exp34'
PARENT = 'RD-1.2-exp12'
EXP33 = ROOT / 'outputs/methods/rd12_exp33_adjacent_context_20261015_r2'
PROFILES = {'llava8': 8, 'qwen32': 32}
CELLS = PREVIOUS.CELLS
CODE_FILES = ('src/videoqa_methods/low_loss_replacement.py',
              'src/videoqa_audit/low_loss_experiment.py',
              'src/videoqa_audit/low_loss_development.py',
              'scripts/run_low_loss_replacement.py')
CASE_PANEL = (('videomme', 'llava8', '012-3'),
              ('videomme', 'llava8', '566-3'),
              ('videomme', 'llava8', '393-2'),
              ('videomme', 'llava8', '386-1'),
              ('videomme', 'qwen32', '384-3'),
              ('mlvu', 'llava8', 'mlvu-dev-Q624'),
              ('mlvu', 'qwen32', 'mlvu-dev-Q866'),
              ('mlvu', 'qwen32', 'mlvu-dev-Q1333'))


def frozen_output(path, value):
    """接口：完整恢复只能核对原离线产物；不静默覆盖已发布选择。"""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen output differs: ' + str(path))
    else:
        durable(path, value)


def prepare_offline(root):
    """接口：全9748项只读重放、单因素选择及事实目录追踪，无任何模型调用。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive(root / 'offline.lock'):
        prior_summary = read_json(EXP33 / 'offline_summary.json')
        if prior_summary['questions'] != 9748 or prior_summary['new_model_calls'] != 0:
            raise ValueError('Frozen exp33 population differs')
        from videoqa_audit.adjacent_context_experiment import verify_offline
        verify_offline(EXP33)
        protocol = root / 'offline_protocol.json'
        code = {p: sha256(ROOT / p) for p in CODE_FILES}
        signature = dict(method=METHOD, parent=PARENT, fixed_proposal='RD-1.2-exp33',
                         prior_summary_sha256=sha256(EXP33 / 'offline_summary.json'),
                         code=code, reference='all original coarse frames equally weighted',
                         maximum_real_calls=626, maximum_synthetic_calls=8)
        if protocol.exists():
            if {k: v for k, v in read_json(protocol).items() if k != 'utc'} != signature:
                raise ValueError('Offline protocol changed')
        else:
            frozen_output(protocol, dict(signature, utc=utc()))
            for name in CODE_FILES:
                dest = root / 'code_snapshot' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)

        # 步骤1：复用原人口、池与特征身份；问题答案及人工事实不进入选择器。
        facts = read_json(FACTS)['entries']
        if len(facts) != 75 or len({f['question_id'] for f in facts}) != 42:
            raise ValueError('Frozen fact denominator differs')
        old_exports = prior_summary['exports']
        exports, stats, witness_paths, alarms = {}, {}, [], []
        started = time.perf_counter()
        for dataset in ('videomme', 'mlvu'):
            for profile, K in PROFILES.items():
                ids = read_json(EXP33 / 'development_ids.json')[dataset]
                development = set(ids)
                records = sorted((EXP33 / 'offline' / dataset / profile).glob('*.json'))
                expected_count = 2700 if dataset == 'videomme' else 2174
                if len(records) != expected_count:
                    raise ValueError('Frozen exp33 file count differs')
                counters = Counter()
                for ordinal, previous in enumerate(records, 1):
                    prior_rel = str(previous.relative_to(EXP33))
                    if sha256(previous) != old_exports[prior_rel]:
                        raise ValueError('Frozen exp33 selection hash differs')
                    old = read_json(previous)
                    q = old['question_id']
                    pool = read_json(old['pool_source'])
                    parent = read_json(old['parent_result'])
                    feature_file = np.load(old['feature_source'], mmap_mode='r', allow_pickle=False)
                    features = (feature_file[pool['original_candidate_indices']]
                                if dataset == 'videomme' and profile == 'llava8'
                                else feature_file)
                    lookup = {source_key(f): i for i, f in enumerate(pool['candidates'])}
                    old_selected = [lookup[source_key(f)]
                                    for f in parent['selection']['selected_frames']]
                    final, trace = select_low_loss_replacement(
                        pool, features, old_selected, old['trace'], K)
                    # 步骤2：复制父版元数据，只按源身份增删一帧；未触发时字节级等于父选择。
                    selection = copy.deepcopy(parent['selection'])
                    if trace['changed']:
                        donor_key = source_key(pool['candidates'][trace['chosen_donor_index']])
                        kept = [f for f in selection['selected_frames']
                                if source_key(f) != donor_key]
                        if len(kept) != len(old_selected) - 1:
                            raise ValueError('Chosen donor not found exactly once')
                        kept.append(copy.deepcopy(pool['candidates'][trace['added_index']]))
                        selection['selected_frames'] = sorted(kept, key=source_key)
                    validate_frames(selection, K)
                    if [lookup[source_key(f)] for f in selection['selected_frames']] != final:
                        raise ValueError('Final source frame mapping differs')
                    if trace['changed'] != old['trace']['changed']:
                        raise ValueError('Frozen exp33 trigger changed')

                    # 步骤3：按原选择单元检查配额与Qwen时间伙伴变化。
                    cells = coarse_selection_cells(pool['plan'], pool['video'])
                    def owner(index):
                        timestamp = source_time(pool['candidates'][index], pool['video'])
                        matches = [r['region_id'] for r in cells['regions']
                                   if Fraction(r['left_fraction']) <= timestamp <= Fraction(r['right_fraction'])]
                        if len(matches) != 1:
                            raise ValueError('Source candidate has ambiguous region owner')
                        return matches[0]
                    before, after = Counter(map(owner, old_selected)), Counter(map(owner, final))
                    if any(after[x] < 1 for x in before) or set(before) != set(after):
                        raise ValueError('Region anchor allocation was removed')
                    group_pairs = lambda indices: [[pool['candidates'][i]['source_pts'] for i in indices[t:t+2]]
                                                   for t in range(0, len(indices), 2)]
                    prior_sources = old['source_sha256']
                    for path in (old['parent_result'], old['pool_source'], old['feature_source']):
                        if path not in prior_sources:
                            raise ValueError('Prior source fingerprint absent: ' + path)
                    record = dict(method=METHOD, parent=PARENT, fixed_proposal='RD-1.2-exp33',
                                  dataset=dataset, profile=profile, question_id=q,
                                  task_type=old['task_type'], stratum=old['stratum'],
                                  selection=selection, parent_result=old['parent_result'],
                                  exp33_offline=str(previous), pool_source=old['pool_source'],
                                  feature_source=old['feature_source'],
                                  source_sha256={str(previous): old_exports[prior_rel],
                                                 old['parent_result']: prior_sources[old['parent_result']],
                                                 old['pool_source']: prior_sources[old['pool_source']],
                                                 old['feature_source']: prior_sources[old['feature_source']]},
                                  trace=trace, parent_region_budgets={str(k): v for k, v in before.items()},
                                  final_region_budgets={str(k): v for k, v in after.items()},
                                  qwen_pairs_before=group_pairs(old_selected) if profile == 'qwen32' else [],
                                  qwen_pairs_after=group_pairs(final) if profile == 'qwen32' else [])
                    target = root / 'offline' / dataset / profile / (q + '.json')
                    frozen_output(target, record)
                    exports[str(target.relative_to(root))] = sha256(target)
                    counters['questions'] += 1
                    counters['changed'] += int(trace['changed'])
                    counters['donor_changed'] += int(trace.get('donor_changed', False))
                    counters['development_changed'] += int(q in development and trace['changed'])
                    if trace['changed']:
                        if trace['added_index'] != old['trace']['winner']['added_index']:
                            raise ValueError('Frozen added frame changed')
                        if trace['coverage_final'] + 1e-6 + 1e-12 < trace['coverage_exp33']:
                            raise ValueError('Coverage bound failed')
                    if dataset == 'videomme':
                        previous_pts = {pool['candidates'][i]['source_pts'] for i in old_selected}
                        new_pts = {pool['candidates'][i]['source_pts'] for i in final}
                        for fact in (x for x in facts if x['question_id'] == q):
                            listed = set(fact['witness_pts'])
                            hit0, hit1 = sorted(previous_pts & listed), sorted(new_pts & listed)
                            item = dict(profile=profile, fact=fact,
                                        before_pts=hit0, after_pts=hit1,
                                        listed_witness_loss=bool(hit0) and not hit1,
                                        listed_witness_gain=not hit0 and bool(hit1))
                            witness_paths.append(item)
                            if item['listed_witness_loss']:
                                alarms.append(item)
                    if ordinal % 100 == 0:
                        durable(root / 'offline_status.json', dict(dataset=dataset, profile=profile,
                            completed=ordinal, total=len(records), elapsed_seconds=time.perf_counter()-started,
                            new_model_calls=0, utc=utc()))
                stats[dataset + '/' + profile] = dict(counters)

        if len(witness_paths) != 150:
            raise ValueError('Expected 75 facts per Video-MME profile')
        frozen_output(root / 'development_ids.json', read_json(EXP33 / 'development_ids.json'))
        frozen_output(root / 'witness_paths.json', witness_paths)
        frozen_output(root / 'witness_alarms.json', alarms)
        cases = []
        for dataset, profile, q in CASE_PANEL:
            old = read_json(EXP33 / 'offline' / dataset / profile / (q + '.json'))
            current = read_json(root / 'offline' / dataset / profile / (q + '.json'))
            cases.append(dict(dataset=dataset, profile=profile, question_id=q,
                              exp33_added_index=(old['trace']['winner'] or {}).get('added_index'),
                              exp33_removed_index=(old['trace']['winner'] or {}).get('replaced_index'),
                              exp34_removed_index=current['trace'].get('chosen_donor_index'),
                              changed_delete=current['trace'].get('donor_changed', False),
                              coverage_parent=current['trace'].get('coverage_parent'),
                              coverage_exp33=current['trace'].get('coverage_exp33'),
                              coverage_final=current['trace'].get('coverage_final')))
        frozen_output(root / 'fixed_cases.json', cases)
        for name in ('development_ids.json', 'witness_paths.json', 'witness_alarms.json', 'fixed_cases.json'):
            exports[name] = sha256(root / name)
        summary = dict(status='offline_complete', method=METHOD, questions=9748,
                       statistics=stats, listed_witness_alarms=len(alarms),
                       new_model_calls=0, protocol_sha256=sha256(protocol),
                       exports=exports, elapsed_seconds=time.perf_counter()-started,
                       utc=utc())
        frozen_output(root / 'offline_summary.json', summary)
        return summary


def verify_offline(root):
    """接口：恢复只核验历史提议、当前源码和冻结导出，不重新选择。"""
    root = Path(root)
    summary = read_json(root / 'offline_summary.json')
    protocol = read_json(root / 'offline_protocol.json')
    if sha256(root / 'offline_protocol.json') != summary['protocol_sha256']:
        raise ValueError('Offline protocol changed')
    for path, old in protocol['code'].items():
        if sha256(ROOT / path) != old:
            raise ValueError('Offline code changed: ' + path)
    if sha256(EXP33 / 'offline_summary.json') != protocol['prior_summary_sha256']:
        raise ValueError('Historical exp33 offline data changed')
    for rel, expected in summary['exports'].items():
        if sha256(root / rel) != expected:
            raise ValueError('Frozen offline output changed: ' + rel)
    return summary
