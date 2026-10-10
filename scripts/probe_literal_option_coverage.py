"""完整八题文字词组覆盖试点：复用冻结OCR，双配置原配额，新增模型及QA0。"""
import argparse
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from videoqa_full.state import rows, durable, exclusive, utc
from videoqa_runtime.common import read_json, sha256
from videoqa_methods.literal_option_coverage import option_word_groups, literal_matches, select_literal_coverage
from videoqa_audit.framework_contraction import BASE

SOURCE = ROOT / 'outputs/analysis/spatial_text__pilot8__20261009_r3'
OCR = ROOT / 'outputs/methods/rd12_exp29_text_state_20261009_r3'
FILES = ('scripts/probe_literal_option_coverage.py', 'src/videoqa_methods/literal_option_coverage.py',
    'src/videoqa_methods/text_state_redundancy.py', 'src/videoqa_methods/weighted_seed_support.py')


def verify(root):
    """接口：原OCR、题选项、特征及父结果变化暂停，零调用恢复保持相同输出。"""
    spec = read_json(root / 'protocol.json')
    for path, expected in spec['sources'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    data = {r['question_id']: dict(question=r['question'], options=r['options']) for r in rows()}
    if any(spec['public_inputs'][q] != data[q] for q in spec['question_ids']):
        raise ValueError('Public question/options changed')
    return spec


def prepare(root):
    """接口：八题全池、双配置及字面匹配规则先冻结，人工事实不进入匹配或选帧。"""
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        return verify(root)
    prior = read_json(SOURCE / 'protocol.json'); data = {r['question_id']: r for r in rows()}
    sources = {str(SOURCE / 'protocol.json'): sha256(SOURCE / 'protocol.json')}; banks = {}
    # 步骤1：OCR缓存绑定原粗细池及逐源返回；不把别题或别版本词集合混入。
    for q in prior['pilot_ids']:
        path = OCR / 'ocr_banks' / (q + '.json'); bank = read_json(path)
        pool = BASE / 'pools' / (q + '.json')
        if bank['pool_sha256'] != sha256(pool) or bank['protocol_sha256'] != sha256(OCR / 'protocol.json'):
            raise ValueError('OCR bank source differs')
        for source, expected in bank['calls'].items():
            if sha256(source) != expected:
                raise ValueError('Original OCR return changed')
            sources[source] = expected
        sources[str(path)] = sha256(path); sources[str(pool)] = sha256(pool); banks[q] = str(path)
    for item in prior['inputs'].values():
        for key in ('pool', 'features', 'parent_trace', 'parent_selection'):
            sources[item[key]] = sha256(item[key])
    # 步骤2：仅采用原公开选择文本，词序关系和数值语义不伪装成完整见证。
    for name in FILES:
        sources[str(ROOT / name)] = sha256(ROOT / name)
        destination = root / 'code_snapshot' / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    spec = dict(parent='RD-1.2-exp07', question_ids=prior['pilot_ids'], inputs=prior['inputs'], banks=banks,
        public_inputs={q: dict(question=data[q]['question'], options=data[q]['options']) for q in prior['pilot_ids']},
        sources=sources, matching='Whole option token-set subset of unchanged OCR words; word order unproven',
        novelty='Unseen matched group fraction within original selection unit; no free fusion coefficient',
        original_L_and_quotas=True, no_new_model_calls=True, final_QA_limit=0,
        gate='Cross-video visible evidence gain before wider preparation; track loss and unsupported relations')
    durable(root / 'protocol.json', spec)
    return spec


def execute(root):
    """接口：匹配→空组控制重放→新选择；所有16组同池预算闭合，没有新增评分。"""
    spec = prepare(root); reports = []; artifacts = {}; started = time.perf_counter()
    # 步骤1：同题OCR跨配置通过源号/PTS映射，原始词序未保留时不宣称有序短语匹配。
    for name, item in spec['inputs'].items():
        profile, q = name.split(':'); pool = read_json(item['pool']); bank = read_json(spec['banks'][q])
        mapping = item['global_mapping']; states = [bank['states'][i] for i in mapping]
        if [[f['source_frame_index'], f['source_pts']] for f in pool['candidates']] != [bank['source_keys'][i] for i in mapping]:
            raise ValueError('OCR source identity differs')
        f = np.load(item['features'], allow_pickle=False)[mapping]
        parent = read_json(item['parent_trace']); parent = parent['exp07'] if item['budget'] == 8 else parent
        public = spec['public_inputs'][q]; groups = option_word_groups(public['question'], public['options'])
        matches = literal_matches(states, groups)
        control, _ = select_literal_coverage(pool, f, parent, [set() for _ in states], item['budget'])
        expected = read_json(item['parent_selection'])['selection']['selected_frames']
        if [pool['candidates'][i] for i in control] != expected:
            raise ValueError('Empty literal group does not replay parent')
        # 步骤2：完整候选竞争不因无匹配缩池；每次实体首次覆盖及其原字可追溯。
        selected, trace = select_literal_coverage(pool, f, parent, matches, item['budget'])
        value = dict(question_id=q, profile=profile, parent='RD-1.2-exp07',
            selection=dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in selected]),
            selected_indices=selected, groups=[list(group) for group in groups], matched_group_indices=[sorted(s) for s in matches],
            matched_source_words={str(i): states[i] for i, m in enumerate(matches) if m},
            trace=trace, new_model_calls=0, final_QA_calls=0, old_control_exact=True)
        path = root / 'selections' / profile / (q + '.json')
        if path.exists() and read_json(path) != value:
            raise ValueError('Completed literal selection changed')
        durable(path, value); artifacts[str(path)] = sha256(path)
        reports.append(dict(question_id=q, profile=profile, groups=groups,
            matching_frames=sum(bool(m) for m in matches), selected_matching_frames=sum(bool(matches[i]) for i in selected),
            changed_frames=len(set(control) ^ set(selected)) // 2))
    # 步骤3：结果只代表选帧，恢复不更新旧实际耗时或把历史答案当新预测。
    summary = dict(reports=reports, artifacts=artifacts, model_calls=0, final_QA_calls=0,
        cached_CPU_seconds=time.perf_counter() - started, scope='Complete eight-question OCR pools, sixteen selections, no accuracy')
    if (root / 'summary.json').exists():
        if read_json(root / 'summary.json')['artifacts'] != artifacts:
            raise ValueError('Completed output hashes changed')
        durable(root / 'restore_audit.json', dict(new_model_calls=0, new_QA_calls=0, utc=utc()))
    else:
        durable(root / 'summary.json', summary)
    print(dict(completed=len(reports), matching_questions=len({r['question_id'] for r in reports if r['matching_frames']}), model_calls=0))


def main():
    """接口：独立CPU试点，不注册正式版本或自动生成开发题回答。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    root = ROOT / 'outputs/analysis' / args.run_id
    with exclusive(root / 'coordinator.lock'):
        execute(root)


if __name__ == '__main__':
    main()
