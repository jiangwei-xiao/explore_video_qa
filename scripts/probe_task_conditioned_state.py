"""问题条件状态先行验证：原定位/配额不动，固定真实竞争银行，最终QA0。"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from videoqa_full.state import rows, durable, exclusive, utc
from videoqa_runtime.common import read_json, sha256
from videoqa_audit.framework_contraction import BASE
from videoqa_budget32.frames import FrameProvider
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.task_conditioned_state import (
    QuestionStateEncoder, image_residual, conditional_redundancy,
)

SPATIAL = ROOT / 'outputs/analysis/spatial_text__pilot8__20261009_r3'
MODEL = Path('/home/models/Qwen/Qwen2.5-VL-7B-Instruct')
TARGETS = {'007-2': [27], '022-1': [25], '035-2': [55, 56, 59],
    '584-3': [520, 582], '191-3': [6], '538-1': [], '103-1': [], '383-2': []}
FILES = ('scripts/probe_task_conditioned_state.py', 'src/videoqa_methods/task_conditioned_state.py',
    'src/videoqa_full/qwen_backend.py', 'src/videoqa_full/qwen_fa2_backend.py',
    'src/videoqa_budget32/frames.py', 'src/videoqa_methods/weighted_seed_support.py')


def key(frame):
    """接口：真实源帧号与PTS共同标识；跨池候选号不能充当源身份。"""
    return frame['source_frame_index'], frame['source_pts']


def evaluation_pairs(pool, parent, failed, question_id):
    """接口：先冻结状态见证与实际竞争者；题号/已知时间仅用于审核，不传给编码器。"""
    targets = list(TARGETS.get(question_id, []))
    # 步骤1：指定新增约束、已保护叉子/垃圾桶与船图；细帧也按精确源记录对应。
    ranges = {'538-1': [(8.75, 8.8)], '103-1': [(4.75, 4.8), (5.75, 5.8), (7.25, 7.3)],
        '383-2': [(942.7, 942.9)]}
    targets += [i for i, frame in enumerate(pool['candidates'])
        if any(lo <= frame['timestamp_seconds'] <= hi for lo, hi in ranges.get(question_id, []))]
    cases = []
    for target in sorted(set(targets)):
        if target >= len(pool['candidates']):
            continue
        region = next(r for r in parent['regions'] if target in r['members'])
        quota = parent['region_budgets'][str(region['region_id'])]
        position = region['prefix'].index(target) if target in region['prefix'] else None
        selected = position is not None and position < quota
        role = 'protected_state' if selected else 'missing_state_opportunity'
        if question_id == '584-3' and target == 582:
            role = 'alternative_lying_witness_not_new_fact'
        if quota == 1 or position == 0:
            cases.append(dict(target=target, role=role, quota=quota,
                status='Fixed anchor or no replacement slot', parent_selected=selected))
            continue
        # 步骤2：保护对照在原入选当时比较下一候选；缺失对照用父实际末位胜者。
        slot = position if selected else quota - 1
        before = region['prefix'][:slot]
        if selected:
            other = next(r for r in failed['regions'] if r['region_id'] == region['region_id'])
            winner = other['prefix'][slot]
            if winner in before or winner == target:
                winner = region['prefix'][slot + 1] if slot + 1 < len(region['prefix']) else None
        else:
            winner = region['prefix'][slot]
        if winner is None:
            cases.append(dict(target=target, role=role, quota=quota, status='No frozen challenger', parent_selected=selected))
            continue
        cases.append(dict(target=target, winner=winner, selected_before=before, role=role,
            quota=quota, region_id=region['region_id'], parent_selected=selected))
    return cases


def verify(root):
    """接口：输入银行、问题、模型和代码变化时暂停，不重复未知前向。"""
    spec = read_json(root / 'protocol.json')
    for path, expected in spec['sources'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    return spec


def prepare(root):
    """接口：先冻结银行及竞争上下文，人工目标仅作评价，不进入问题编码。"""
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        return verify(root)
    prior = read_json(SPATIAL / 'protocol.json'); original_rows = {r['question_id']: r for r in rows()}
    sources = {str(SPATIAL / 'protocol.json'): sha256(SPATIAL / 'protocol.json')}
    banks = {}; profiles = {}
    # 步骤1：并集含父种子、父输入、失败文字版本实际竞争帧及固定见证，不按新分数挑图。
    for q in prior['pilot_ids']:
        pp = BASE / 'pools' / (q + '.json'); base = read_json(pp)
        lookup = {key(f): i for i, f in enumerate(base['candidates'])}; bank_ids = set()
        for profile in prior['profiles']:
            item = prior['inputs'][profile + ':' + q]; pool = read_json(item['pool'])
            parent = read_json(item['parent_trace'])
            if item['budget'] == 8:
                parent = parent['exp07']
            trace_path = SPATIAL / 'traces' / profile / (q + '.json')
            spatial = read_json(trace_path); membership = [lookup[key(f)] for f in pool['candidates']]
            selected = [i for region in parent['regions'] for i in region['prefix'][:parent['region_budgets'][str(region['region_id'])]]]
            bank_ids.update(membership[i] for i in selected)
            bank_ids.update(membership[i] for i in pool['plan']['seed_indices'])
            bank_ids.update(membership[i] for i in spatial['selected_indices'])
            comparisons = evaluation_pairs(pool, parent, spatial, q)
            for case in comparisons:
                bank_ids.add(membership[case['target']])
                if 'winner' in case:
                    bank_ids.add(membership[case['winner']])
                    bank_ids.update(membership[j] for j in case['selected_before'])
            for path in (item['pool'], item['features'], item['parent_trace'], item['parent_selection'], trace_path):
                sources[str(path)] = sha256(path)
            profiles[profile + ':' + q] = dict(item, global_mapping=membership,
                spatial_trace=str(trace_path), parent_selected_indices=selected,
                evaluation_pairs=comparisons)
        bank_ids = sorted(bank_ids, key=lambda i: (base['candidates'][i]['source_pts'], i))
        frames = [base['candidates'][i] for i in bank_ids]; hashes = []; shape = None
        sources[str(pp)] = sha256(pp); sources[base['video']['path']] = sha256(base['video']['path'])
        # 步骤2：小批精确PTS解码保存无损试点图，最多768张，不保存全片候选PNG。
        for start in range(0, len(frames), 16):
            images, rgb_hashes = FrameProvider(dict(video=base['video'], selected_frames=frames[start:start + 16])).decode()
            for ordinal, image in enumerate(images, start):
                current_shape = list(np.asarray(image).shape)
                if shape is None:
                    shape = current_shape
                if current_shape != shape:
                    raise ValueError('One-question blank baseline requires identical source geometry')
                path = root / 'bank' / q / f'{ordinal:03d}.png'; path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path); image.close(); sources[str(path)] = sha256(path)
            hashes.extend(rgb_hashes)
        banks[q] = dict(global_indices=bank_ids, frames=frames, rgb_sha256=hashes,
            rgb_shape=shape, question=original_rows[q]['question'], count=len(frames))
        durable(root / 'status.json', dict(status='preparing', questions=len(banks), total=8, utc=utc()))
        print('Prepared bank', q, len(frames), flush=True)
    count = sum(b['count'] for b in banks.values())
    if count > 768:
        raise ValueError('Frozen pilot source budget exceeded')
    # 步骤3：身份包括完整原权重及固定实现，语言prefill与最终问答分开记账。
    for name in FILES:
        path = ROOT / name; sources[str(path)] = sha256(path)
        destination = root / 'code_snapshot' / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for path in MODEL.iterdir():
        if path.is_file() and path.suffix in ('.json', '.safetensors'):
            sources[str(path)] = sha256(path)
    official = Path('/App/conda/envs/explore_video_qa_lmms_qwen/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py')
    sources[str(official)] = sha256(official)
    spec = dict(parent='RD-1.2-exp07', banks=banks, profiles=profiles, sources=sources,
        language_feature_rule='Normalize(last-input hidden state with frame minus same-question same-grid blank frame state)',
        changed_quantity='Only repetition D; original BLIP L, anchors and exp07 quotas fixed',
        targets_for_evaluation=TARGETS, real_prefill_limit=count, blank_baseline_limit=8,
        synthetic_prefill_limit=4, final_QA_limit=0, input_options_or_answers=False,
        pilot_gate='Two distinct missing-state videos improve against actual parent winner; no confirmed protected-state priority loss; visual review before wider selection',
        development_gate='Only after full-pool pilot protection: both >= exp07 and (Qwen>=140 or LLaVA>=128)',
        representation_status='Unproven question-conditioned proxy; no probability, semantic-distance or optimality claim')
    durable(root / 'protocol.json', spec)
    return spec


def feature_call(root, relative, image, question, encoder, category):
    """接口：调用前开始记账、返回即保存原状态；未知前向禁止重发，完成项只读取。"""
    path = root / 'calls' / relative
    if path.exists():
        value = read_json(path)
        if value['status'] != 'returned' or value['question'] != question or value['category'] != category:
            raise ValueError('Uncertain or mismatched prefill; automatic retry forbidden')
        actual = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()
        if actual != value['metadata']['rgb_sha256']:
            raise ValueError('Cached source RGB differs')
        return np.asarray(value['state'], dtype=np.float32), value['metadata']
    def started(metadata):
        """接口：模型前向前写入输入身份；连接中断不能被解释为尚未调用。"""
        durable(path, dict(status='started', question=question, category=category,
            metadata=metadata, pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), utc=utc()))
    state, metadata = encoder.encode(image, question, before_forward=started)
    # 步骤1：原浮点状态先存为返回检查点；后续归一化失败不触发再次前向。
    durable(path, dict(status='returned', question=question, category=category,
        state=state.tolist(), metadata=metadata, utc=utc()))
    return state, metadata


def extract(root, spec):
    """接口：单GPU长期进程提取有界银行，4合成/8空白基线与真实帧成本分开。"""
    from PIL import Image
    encoder = QuestionStateEncoder(); encoder.torch.set_num_threads(2)
    durable(root / 'environment.json', encoder.load_report)
    # 步骤1：合成只检验接口与重复性，无真实答案；不选择层、池化或融合参数。
    synthetics = [((1280, 720), (20, 80, 140)), ((720, 1280), (20, 80, 140)),
        ((1280, 720), (20, 80, 140)), ((1280, 720), (20, 80, 140))]
    synthetic_states = []
    for i, (size, color) in enumerate(synthetics):
        image = Image.new('RGB', size, color)
        state, _ = feature_call(root, f'synthetic/{i}.json', image,
            'What color is visible?' if i < 3 else 'What shapes are visible?', encoder, 'synthetic')
        synthetic_states.append(state); image.close()
    if not np.array_equal(synthetic_states[0], synthetic_states[2]):
        raise ValueError('Identical synthetic input did not repeat exactly')
    if np.array_equal(synthetic_states[0], synthetic_states[3]):
        raise ValueError('Changing the question did not change the observed state')
    done = 0; results = {}
    # 步骤2：空白媒体与真实媒体保持同形状、问题和编码位置，避免无图长度变化。
    for q, bank in spec['banks'].items():
        height, width, _ = bank['rgb_shape']; blank = Image.new('RGB', (width, height), 'black')
        baseline, baseline_meta = feature_call(root, f'blank/{q}.json', blank,
            bank['question'], encoder, 'blank_baseline'); blank.close()
        vectors = {}; valid = {}; norms = {}; costs = []
        for ordinal, index in enumerate(bank['global_indices']):
            with Image.open(root / 'bank' / q / f'{ordinal:03d}.png') as image:
                state, meta = feature_call(root, f'real/{q}/{ordinal:03d}.json', image,
                    bank['question'], encoder, 'real_frame')
            if meta['rgb_sha256'] != bank['rgb_sha256'][ordinal] or meta['token_sha256'] != baseline_meta['token_sha256']:
                raise ValueError('Source or same-question encoding positions differ')
            vector, is_valid, norm = image_residual(state, baseline)
            vectors[str(index)] = vector.tolist(); valid[str(index)] = is_valid; norms[str(index)] = norm
            costs.append(meta['forward_seconds']); done += 1
            durable(root / 'status.json', dict(status='extracting', completed=done,
                total=spec['real_prefill_limit'], active_question=q, pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        value = dict(vectors=vectors, valid=valid, residual_norms=norms,
            actual_forward_seconds=sum(costs), original_question=bank['question'])
        durable(root / 'features' / (q + '.json'), value); results[q] = value
        print('Encoded bank', q, len(vectors), 'seconds', round(sum(costs), 2), flush=True)
    return results


def evaluate(root, spec, representations):
    """接口：固定父实际选帧上下文，比较所需状态与实际末位胜者，不能代替完整新输入。"""
    reports = []
    # 步骤1：原BLIP支持完整复用；仅D消费新状态，人工标注只决定检查哪张画面。
    for name, item in spec['profiles'].items():
        profile, q = name.split(':'); pool = read_json(item['pool'])
        parent = read_json(item['parent_trace']); parent = parent['exp07'] if item['budget'] == 8 else parent
        old = np.load(item['features'], allow_pickle=False)[item['global_mapping']]
        rep = representations[q]; mapping = item['global_mapping']
        for case in item['evaluation_pairs']:
            target = case['target']
            if 'winner' not in case:
                reports.append(dict(question_id=q, profile=profile, **case))
                continue
            region = next(r for r in parent['regions'] if target in r['members'])
            quota = case['quota']; before = case['selected_before']; winner = case['winner']
            plan = next(r for r in pool['plan']['regions'] if r['region_id'] == region['region_id'])
            support, _, _ = weighted_support(old, region['members'], plan['seed_indices'], pool['initial_scores'])
            support_lookup = dict(zip(region['members'], support))
            values = {}
            for label, index in [('target', target), ('winner', winner)]:
                global_index = str(mapping[index]); context_indices = [str(mapping[j]) for j in before]
                new_d = conditional_redundancy(old[index], old[before], rep['vectors'][global_index],
                    np.asarray([rep['vectors'][j] for j in context_indices]), rep['valid'][global_index],
                    [rep['valid'][j] for j in context_indices])
                old_d = float(np.clip(old[before] @ old[index], 0, 1).max()) if before else 0.
                values[label] = dict(old_score=float(support_lookup[index] * np.sqrt(1 - old_d)),
                    new_score=float(support_lookup[index] * np.sqrt(1 - new_d)), old_D=old_d, new_D=new_d)
            reports.append(dict(question_id=q, profile=profile, target=target, winner=winner,
                target_frame=pool['candidates'][target], winner_frame=pool['candidates'][winner],
                selected_before=before, quota=quota, role=case['role'], target_parent_selected=case['parent_selected'],
                values=values, new_target_ahead=bool(values['target']['new_score'] > values['winner']['new_score'] + 1e-6)))
    # 步骤2：声明范围与成本；不据固定上下文的排序直接附用旧答案或报告精度。
    calls = [read_json(p) for p in (root / 'calls').rglob('*.json')]
    if any(c['status'] != 'returned' for c in calls):
        raise ValueError('Uncertain prefill remains')
    summary = dict(reports=reports, actual_prefill_calls=len(calls),
        real_frame_calls=sum(c['category'] == 'real_frame' for c in calls),
        synthetic_calls=sum(c['category'] == 'synthetic' for c in calls),
        blank_baseline_calls=sum(c['category'] == 'blank_baseline' for c in calls),
        forward_seconds=sum(c['metadata']['forward_seconds'] for c in calls),
        artifacts={str(p): sha256(p) for folder in ['calls', 'features'] for p in (root / folder).rglob('*.json')},
        final_QA_calls=0, scope='Fixed-context state competition; no new full-pool selection or accuracy',
        next_stage='Requires numerical and semantic gate review, not automatically launched')
    if summary['real_frame_calls'] != spec['real_prefill_limit'] or summary['synthetic_calls'] > 4 or summary['blank_baseline_calls'] != 8:
        raise ValueError('Frozen prefill budget mismatch')
    durable(root / 'summary.json', summary)
    durable(root / 'status.json', dict(status='completed', pid=os.getpid(), utc=utc()))
    return summary


def verify_completed(root):
    """接口：完整返回/特征字节不变才接受完成恢复，零加载/零前向，不改旧总结。"""
    value = read_json(root / 'summary.json')
    # 步骤1：检查完整结果哈希，不把已存在但被修改的返回当作可直接复用。
    for path, expected in value['artifacts'].items():
        if sha256(path) != expected:
            raise ValueError('Completed prefill artifact changed: ' + path)
    # 步骤2：单列只读恢复证明，不覆盖真实成本、旧答案或完成时间。
    durable(root / 'completed_restore_audit.json', dict(status='passed',
        artifacts=len(value['artifacts']), new_model_forwards=0, new_QA_calls=0, utc=utc()))
    return value


def main():
    """接口：后台完成准备/特征/先行评价；未知前向暂停，不自动启动200题。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    root = ROOT / 'outputs/diagnostics' / args.run_id
    try:
        with exclusive(root / 'coordinator.lock'):
            durable(root / 'launch.json', dict(pid=os.getpid(), sid=os.getsid(0),
                visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'), utc=utc()))
            spec = prepare(root); verify(root)
            if (root / 'summary.json').exists():
                verify_completed(root)
            else:
                representations = extract(root, spec)
                evaluate(root, spec, representations)
    except BaseException as exc:
        durable(root / 'failure.json', dict(status='paused', error=repr(exc), utc=utc()))
        raise


if __name__ == '__main__':
    main()
