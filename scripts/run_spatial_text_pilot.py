"""同位文字差异八题试点：复用冻结OCR，双配置固定父配额，新增模型调用为0。"""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit.framework_contraction import BASE, locations
from videoqa_audit.text_state_experiments import PILOT
from videoqa_methods.spatial_text_redundancy import layout_state, select_spatial_text

LEGACY = ROOT / 'outputs/methods/rd12_exp29_text_state_20261009_r3'
FILES = ('scripts/run_spatial_text_pilot.py', 'src/videoqa_methods/spatial_text_redundancy.py',
    'src/videoqa_methods/weighted_seed_support.py', 'src/videoqa_methods/density_selection.py')
TARGETS = {'007-2': [27], '022-1': [25], '035-2': [55, 56, 59], '584-3': [582], '191-3': [6]}


def source_layouts(question, root):
    """接口：已完成同源OCR的原框/原字→空间状态；不再次OCR，不消费标准答案。"""
    bank_path = LEGACY / 'ocr_banks' / (question + '.json'); bank = read_json(bank_path)
    pool_path = BASE / 'pools' / (question + '.json'); pool = read_json(pool_path)
    if bank['pool_sha256'] != sha256(pool_path): raise ValueError('OCR source pool changed')
    if bank['protocol_sha256'] != sha256(LEGACY / 'protocol.json'):
        raise ValueError('OCR source protocol differs')
    for path, expected in bank['calls'].items():
        if sha256(path) != expected: raise ValueError('Frozen OCR return changed')
    vid = read_json(LEGACY / 'protocol.json')['question_video_ids'][question]
    states = []; sources = {str(bank_path): sha256(bank_path)}
    # 步骤1：每张原帧实际尺寸来自旧RPC账本，坐标不是按平均FPS或统一尺寸推断。
    for frame in pool['candidates']:
        path = LEGACY / 'frame_calls' / vid / (str(frame['source_pts']) + '.json')
        call = read_json(path)
        if call['status'] != 'completed' or call['source_pts'] != frame['source_pts'] or call['source_frame_index'] != frame['source_frame_index']:
            raise ValueError('OCR source-frame identity differs')
        state = layout_state(call['ocr']['regions'], call['rgb_shape'])
        states.append(state); sources[str(path)] = sha256(path)
    # 步骤2：空间解析是旧资产的新展示/计算，原识别文字和模型返回一字不改。
    path = Path(root) / 'layouts' / (question + '.json')
    value = dict(question_id=question, states=states, source_keys=bank['source_keys'], sources=sources,
        original_rgb_shapes='Per-source rgb_shape from completed OCR ledger', new_model_calls=0)
    durable(path, value)
    return value, sources


def prepare(root):
    """接口：先冻结八题双配置的同池映射、原配额及单一差异规则；不根据新选择挑题。"""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists(): return verify(root)
    sources = {str(LEGACY / 'protocol.json'): sha256(LEGACY / 'protocol.json')}
    inputs = {}; exports = {}
    # 步骤1：同一源帧跨K32/K8共享OCR，按历史全局候选映射而非局部编号猜测。
    for q in PILOT:
        layout, source_hashes = source_layouts(q, root); sources.update(source_hashes)
        exports[q] = sha256(root / 'layouts' / (q + '.json'))
        for profile, k in [('qwen32', 32), ('llava8', 8)]:
            pp, fp, tp, sp = locations(q, k)
            pool = read_json(pp); vectors = np.load(fp, allow_pickle=False)
            mapping = list(range(len(vectors))) if k == 32 else pool['original_candidate_indices']
            if [[f['source_frame_index'], f['source_pts']] for f in pool['candidates']] != [layout['source_keys'][i] for i in mapping]:
                raise ValueError('OCR, feature and source mappings differ')
            for p in (pp, fp, tp, sp): sources[str(p)] = sha256(p)
            inputs[profile + ':' + q] = dict(pool=str(pp), features=str(fp), parent_trace=str(tp),
                parent_selection=str(sp), global_mapping=mapping, budget=k)
    # 步骤2：唯一机制只修正D，几何/字符差异是派生量，不增融合系数或查询。
    spec = dict(reference_method='RD-1.2-exp07', pilot_ids=list(PILOT), profiles=['qwen32', 'llava8'],
        inputs=inputs, layout_exports=exports, sources=sources, code_sha256={p: sha256(ROOT / p) for p in FILES},
        rule='Unique mutual spatial correspondence first; D_pair=clip(cos,0,1)*(1-max matched IoU*normalized edit); then max selected',
        source_ocr_confidence=0.5, quotas='parent exp07 fixed', new_model_calls=0, new_QA_calls=0,
        scope='Eight familiar pilot cases, not development accuracy or semantic field certainty')
    for p in FILES:
        dest = root / 'code_snapshot' / p; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / p, dest)
    durable(root / 'protocol.json', spec)
    return spec


def verify(root):
    """接口：旧源、代码、配置及映射变化暂停，零模型恢复不重写冻结布局。"""
    root = Path(root); spec = read_json(root / 'protocol.json')
    if spec['pilot_ids'] != list(PILOT) or spec['new_model_calls'] != 0: raise ValueError('Pilot population changed')
    for p, h in spec['sources'].items():
        if sha256(p) != h: raise ValueError('Source changed: ' + p)
    for p, h in spec['code_sha256'].items():
        if sha256(ROOT / p) != h: raise ValueError('Frozen code changed')
    for q, h in spec['layout_exports'].items():
        if sha256(root / 'layouts' / (q + '.json')) != h: raise ValueError('Frozen layout changed')
    return spec


def execute(root):
    """接口：固定原预算、锚点和L，只比较文字空间修正的新局部前缀。"""
    root = Path(root)
    with exclusive(root / 'coordinator.lock'):
        spec = prepare(root); verify(root); files = {}; reports = []
        # 步骤1：全无文字控制必须复现父选帧，确认不是重建接口引入其他变化。
        for profile in spec['profiles']:
            for q in PILOT:
                item = spec['inputs'][profile + ':' + q]; pool = read_json(item['pool'])
                f = np.load(item['features'], allow_pickle=False)[item['global_mapping']]
                parent = read_json(item['parent_trace']); parent = parent['exp07'] if item['budget'] == 8 else parent
                layouts = read_json(root / 'layouts' / (q + '.json'))['states']
                states = [layouts[i] for i in item['global_mapping']]
                control, _ = select_spatial_text(pool, f, parent, [() for _ in states], item['budget'])
                expected = read_json(item['parent_selection'])['selection']
                frames = lambda ids: [pool['candidates'][i] for i in ids]
                if frames(control) != expected['selected_frames']: raise ValueError('Empty-layout control does not replay parent')
                started = time.perf_counter()
                selected, trace = select_spatial_text(pool, f, parent, states, item['budget'])
                trace_path = root / 'traces' / profile / (q + '.json')
                value = dict(question_id=q, profile=profile, reference_method='RD-1.2-exp07',
                    condition='spatially_matched_text', selection=dict(video=pool['video'], selected_frames=frames(selected)),
                    selected_indices=selected, source_pool=item['pool'], trace_file=str(trace_path.relative_to(root)),
                    new_model_calls=0, original_quotas_fixed=True)
                path = root / 'offline' / profile / (q + '.json')
                if path.exists():
                    if read_json(path) != value or read_json(trace_path) != trace: raise ValueError('Pilot selection changed')
                else:
                    durable(trace_path, trace); durable(path, value)
                files[str(path)] = sha256(path); files[str(trace_path)] = sha256(trace_path)
                # 步骤2：已列状态只在选择之后评价，不传给空间差异或竞争函数。
                targets = []
                for i in TARGETS.get(q, []):
                    a = next(r for r in parent['regions'] if i in r['members'])
                    b = next(r for r in trace['regions'] if i in r['members'])
                    pos = lambda r: r['prefix'].index(i) + 1 if i in r['prefix'] else None
                    targets.append(dict(candidate_index=i, old_position=pos(a), new_position=pos(b),
                        quota=parent['region_budgets'][str(a['region_id'])], selected=i in selected,
                        parent_selected=i in control))
                reports.append(dict(question_id=q, profile=profile, targets=targets,
                    changed_frames=len(set(control) ^ set(selected)) // 2, empty_control_exact=True,
                    local_selection_seconds=time.perf_counter() - started))
                durable(root / 'status.json', dict(status='selecting', completed=len(reports), total=16,
                    pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        # 步骤3：结果/轨迹闭合后完成，耗时只代表缓存条件，不称原片E2E。
        result = dict(status='completed', reports=reports, artifacts=files, new_model_calls=0, new_QA_calls=0,
            semantic_review='pending', utc=utc(), scope=spec['scope'])
        if (root / 'summary.json').exists() and read_json(root / 'summary.json')['artifacts'] != files:
            raise ValueError('Completed artifacts changed on recovery')
        durable(root / 'summary.json', result); durable(root / 'status.json', dict(status='completed', utc=utc()))
        return result


def main():
    """接口：独立零调用试点，不登记正式版本或自动启动200题问答。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum(): parser.error('Invalid run ID')
    root = ROOT / 'outputs/analysis' / args.run_id
    try: execute(root)
    except BaseException as exc:
        durable(root / 'failure.json', dict(status='paused', error=repr(exc), utc=utc())); raise


if __name__ == '__main__':
    main()
