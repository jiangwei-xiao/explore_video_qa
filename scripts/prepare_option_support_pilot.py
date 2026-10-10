"""完整八题合法池的选项支持准备：流式解码，复用同输入概率，最终QA0。"""
import argparse
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from videoqa_full.state import durable, exclusive, utc
from videoqa_runtime.common import read_json, sha256
from videoqa_methods.option_evidence_support import OptionEvidenceScorer, evidence_support
from videoqa_methods.option_evidence_selection import select_option_evidence
from videoqa_methods.density_selection import frame_key
from probe_option_evidence import verify as verify_parent, call as new_call, BANK

PARENT = ROOT / 'outputs/diagnostics/option_support__anchor_pairs8__20261009_r1'


def prepare(root):
    """接口：原八题与同一公式扩大到完整合法候选，新增额度独立冻结，不改原银行。"""
    if (root / 'protocol.json').exists():
        return verify(root)
    prior = verify_parent(PARENT); sources = dict(prior['sources'])
    sources[str(PARENT / 'protocol.json')] = sha256(PARENT / 'protocol.json')
    profiles = {}; candidates = 0; baselines = 0
    # 步骤1：只跳过配额1的等价不变区域，其他合法候选全部需要E，不按分数挑子集。
    for q, item in prior['profiles'].items():
        pool = read_json(item['pool']); parent = read_json(item['parent_trace'])
        active = [region['region_id'] for region in parent['regions'] if parent['region_budgets'][str(region['region_id'])] > 1]
        lookup = {i: region['region_id'] for region in parent['regions'] for i in region['members']}
        anchors = {region['region_id']: region['anchor_index'] for region in pool['plan']['regions']}
        ids = [i for i in range(len(pool['candidates'])) if lookup[i] in active and i != anchors[lookup[i]]]
        profiles[q] = dict(item, active_regions=active, candidate_indices=ids,
            candidate_regions={str(i): lookup[i] for i in ids}, anchors={str(k): v for k, v in anchors.items()})
        candidates += len(ids); baselines += len(active)
    for path in [Path(__file__), ROOT / 'src/videoqa_methods/option_evidence_selection.py']:
        sources[str(path)] = sha256(path)
    spec = dict(parent_method='RD-1.2-exp07', source_probe=str(PARENT), sources=sources,
        profiles=profiles, candidate_positions=candidates, anchor_baseline_positions=baselines,
        maximum_new_auxiliary_forwards=candidates + baselines, final_QA_limit=0,
        rule=prior['local_rule'], evidence=prior['evidence'], budget=32,
        scope='Complete original legal candidate pool; not development or full-dataset QA')
    durable(root / 'protocol.json', spec)
    return spec


def verify(root):
    """接口：所有旧源与当前公式身份一致才恢复；未知辅助前向不得重发。"""
    spec = read_json(root / 'protocol.json')
    for path, expected in spec['sources'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    return spec


def obtain(root, relative, images, public, identity, scorer_holder, category):
    """接口：同像素/公开题选项/源顺序才导入旧概率，导入成本不冒充新推理。"""
    path = root / 'calls' / relative
    hashes = [hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() for image in images]
    if path.exists():
        value = read_json(path)
        if value['status'] != 'returned' or value['public_input'] != public or value['metadata']['rgb_sha256'] != hashes:
            raise ValueError('Uncertain or different full-pool call')
        if value['source_identity']['source_keys'] != identity['source_keys']:
            raise ValueError('Cached physical source order differs')
        return value
    old_path = PARENT / 'calls' / relative
    # 步骤1：银行行号不影响实际模型输入；导入只以物理源、公开文本和RGB判定。
    if old_path.exists():
        old = read_json(old_path)
        if old['status'] != 'returned' or old['public_input'] != public or old['metadata']['rgb_sha256'] != hashes:
            raise ValueError('Same-input probability import differs')
        if old['source_identity']['source_keys'] != identity['source_keys']:
            raise ValueError('Imported source order differs')
        value = dict(old, source_identity=identity,
            imported_probability=dict(path=str(old_path), sha256=sha256(old_path), protocol_sha256=sha256(PARENT / 'protocol.json')))
        durable(path, value)
        return value
    # 步骤2：只在真正缺少返回时加载模型，开始状态由原安全账本函数持久化。
    if not scorer_holder:
        scorer_holder.append(OptionEvidenceScorer()); scorer_holder[0].torch.set_num_threads(2)
        durable(root / 'environment.json', scorer_holder[0].load_report)
    return new_call(root, relative, images, public, scorer_holder[0], category, identity)


def process_question(root, q, item, holder):
    """接口：原视频起点精确计数的流式候选，最多保留原锚点图和当前RGB。"""
    import av
    from PIL import Image
    pool = read_json(item['pool']); parent = read_json(item['parent_trace']); public = item['public_input']
    ordinal = {index: n for n, index in enumerate(item['bank']['global_indices'])}
    anchor_images = {}; baselines = {}; values = {}
    # 步骤1：锚点都来自冻结种子银行；复制自身作为相同输入槽数的基线。
    for rid in item['active_regions']:
        anchor = item['anchors'][str(rid)]; index = item['global_mapping'][anchor]
        image = Image.open(BANK / 'bank' / q / f'{ordinal[index]:03d}.png').convert('RGB')
        anchor_images[rid] = image; frame = pool['candidates'][anchor]
        identity = dict(question_id=q, region_id=rid, anchor=anchor,
            source_indices=[anchor, anchor], source_keys=[[frame['source_frame_index'], frame['source_pts']]] * 2,
            physical_unique_sources=1, scope='Repeated-anchor baseline')
        baselines[rid] = obtain(root, f'anchor_baseline/{q}/{rid}_anchor.json', [image, image],
            public, identity, holder, 'anchor_baseline')
    needed = {pool['candidates'][i]['source_pts']: i for i in item['candidate_indices']}
    try:
        # 步骤2：不按平均FPS推算源号；只将合法候选转换RGB，其他源帧解码后立即释放。
        with av.open(pool['video']['path']) as container:
            stream = container.streams.video[0]; stream.codec_context.thread_count = 2
            if Fraction(stream.time_base) != Fraction(pool['video']['time_base']) or (stream.start_time or 0) != pool['video']['start_pts']:
                raise ValueError('Video clock differs')
            for source_index, decoded in enumerate(container.decode(stream)):
                if decoded.pts not in needed:
                    continue
                i = needed[decoded.pts]; frame = pool['candidates'][i]
                if source_index != frame['source_frame_index']:
                    raise ValueError('Source frame number differs')
                rid = item['candidate_regions'][str(i)]; anchor = item['anchors'][str(rid)]
                image = Image.fromarray(decoded.to_ndarray(format='rgb24'))
                ordered = sorted([anchor, i], key=lambda j: frame_key(pool['candidates'][j]))
                pictures = [anchor_images[rid] if j == anchor else image for j in ordered]
                identity = dict(question_id=q, region_id=rid, anchor=anchor, candidate=i,
                    source_indices=ordered, source_keys=[[pool['candidates'][j]['source_frame_index'], pool['candidates'][j]['source_pts']] for j in ordered],
                    physical_unique_sources=2, scope='Complete original legal pool')
                value = obtain(root, f'candidate_pair/{q}/{rid}_{i}.json', pictures, public, identity, holder, 'candidate_pair')
                baseline = baselines[rid]
                if value['metadata']['token_sha256'] != baseline['metadata']['token_sha256'] or value['metadata']['video_grid_thw'] != baseline['metadata']['video_grid_thw']:
                    raise ValueError('Fixed two-position context differs')
                values[str(i)] = evidence_support(value['probabilities'], baseline['probabilities'])['value']
                image.close()
                durable(root / 'status.json', dict(status='scoring_full_pool', active_question=q,
                    completed_question_candidates=len(values), total_question_candidates=len(needed),
                    pid=os.getpid(), sid=os.getsid(0), utc=utc()))
                if len(values) == len(needed):
                    break
        if len(values) != len(needed):
            raise ValueError('Missing legal source candidate')
    finally:
        for image in anchor_images.values():
            image.close()
    # 步骤3：完整E后才选择；旧E=1控制重放，不借银行选择冒充全池选择。
    f = np.load(item['features'], allow_pickle=False)[item['global_mapping']]
    control, _ = select_option_evidence(pool, f, parent, {i: 1. for i in map(int, values)}, 32)
    expected = sorted(item['parent_selected_indices'], key=lambda j: frame_key(pool['candidates'][j]))
    if control != expected:
        raise ValueError('Full-pool control differs from frozen parent')
    selected, trace = select_option_evidence(pool, f, parent, values, 32)
    durable(root / 'evidence' / (q + '.json'), values)
    durable(root / 'selections' / (q + '.json'), dict(question_id=q, selected_indices=selected,
        selection=dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in selected]),
        trace=trace, old_control_exact=True, full_original_pool=True, final_QA_calls=0))


def main():
    """接口：完整八题准备后台独立运行；未知前向暂停，不自动开启200题生成。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    root = ROOT / 'outputs/diagnostics' / args.run_id
    try:
        with exclusive(root / 'coordinator.lock'):
            spec = prepare(root); verify(root)
            if (root / 'summary.json').exists():
                saved = read_json(root / 'summary.json')
                if any(sha256(path) != expected for path, expected in saved['artifacts'].items()):
                    raise ValueError('Completed full-pool output changed')
                durable(root / 'restore_audit.json', dict(new_model_forwards=0, new_final_QA_calls=0, utc=utc()))
                return
            durable(root / 'launch.json', dict(pid=os.getpid(), sid=os.getsid(0), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), utc=utc()))
            holder = []
            for q, item in spec['profiles'].items():
                process_question(root, q, item, holder)
            calls = [read_json(p) for p in (root / 'calls').rglob('*.json')]
            if len(calls) != spec['candidate_positions'] + spec['anchor_baseline_positions'] or any(c['status'] != 'returned' for c in calls):
                raise ValueError('Full-pool auxiliary call accounting differs')
            summary = dict(completed_questions=len(spec['profiles']), new_auxiliary_forwards=sum('imported_probability' not in c for c in calls),
                imported_probability_calls=sum('imported_probability' in c for c in calls),
                new_model_seconds=sum(c['metadata']['forward_seconds'] for c in calls if 'imported_probability' not in c),
                final_QA_calls=0, artifacts={str(p): sha256(p) for folder in ['calls', 'evidence', 'selections'] for p in (root / folder).rglob('*.json')})
            durable(root / 'summary.json', summary); durable(root / 'status.json', dict(status='completed', utc=utc()))
    except BaseException as error:
        durable(root / 'failure.json', dict(status='paused', error=repr(error), utc=utc())); raise


if __name__ == '__main__':
    main()
