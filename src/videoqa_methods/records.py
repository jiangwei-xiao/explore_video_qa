from pathlib import Path
import math
import numpy as np
from videoqa_runtime.common import read_json, sha256
from videoqa_runtime.baseline_selection import ProtocolError, select_topk
from videoqa_runtime.protocol import parse_answer
from .algorithm import membership

VARIANTS = ('A', 'B', 'C', 'D')


def result_path(run, variant, qid):
    """统一生成每题每变体的结果路径，保证完成检查和续跑使用同一定位规则。"""
    return Path(run) / 'results' / variant / f'{qid}.json'


def validate_result(record, row, fingerprint):
    """验收接口：核对身份、候选、16帧约束、配额、解析计分和计时；运行错误不能被当成完整实验结果。"""
    if record['status'] != 'completed' or record['protocol_sha256'] != fingerprint:
        raise ProtocolError('Method result status or fingerprint mismatch')
    if record['question_id'] != row['question_id'] or record['video_id'] != row['video_id']:
        raise ProtocolError('Method result identity mismatch')
    if record['question'] != row['question'] or record['options'] != row['options']:
        raise ProtocolError('Original question or options changed')
    pool, answer, variant = record['pool'], record['answer'], record['variant']
    if variant not in VARIANTS:
        raise ProtocolError('Unknown method variant')
    rows, scores, indices = pool['candidates'], pool['scores'], pool['selected_indices']
    if len(rows) != len(scores) or [r['candidate_index'] for r in rows] != list(range(len(rows))):
        raise ProtocolError('Invalid candidate pool indices')
    if not np.isfinite(scores).all() or any(not 0 <= r <= 1 for r in scores):
        raise ProtocolError('Invalid retrieval scores')
    if len(set(r['source_pts'] for r in rows)) != len(rows):
        raise ProtocolError('Duplicate source frames in candidate pool')
    if len(indices) != 16 or len(set(indices)) != 16 or pool['selected_frames'] != [rows[i] for i in indices]:
        raise ProtocolError('Invalid 16-frame selection')
    if [r['source_pts'] for r in pool['selected_frames']] != sorted(r['source_pts'] for r in pool['selected_frames']):
        raise ProtocolError('Final frames not in source order')
    if len(rows) - pool['initial_count'] > 24:
        raise ProtocolError('Too many fine candidates')
    if answer['pixel_shape'] != [16, 3, 384, 384] or answer['visual_tokens'] != 3360:
        raise ProtocolError('Unexpected actual visual token layout')
    if answer['prefill_tokens'] != answer['text_input_tokens'] - 1 + 3360:
        raise ProtocolError('Unexpected multimodal truncation')
    expected = parse_answer(answer['raw_output'])
    if answer['parsed_answer'] != expected or record['correct'] != (expected == 'ABCD'[row['answer_index']]):
        raise ProtocolError('Invalid prediction parsing or scoring')
    if record['reference_answer'] != 'ABCD'[row['answer_index']]:
        raise ProtocolError('Reference answer mismatch')
    if variant in ('A', 'B', 'C'):
        if np.bincount(membership(rows, pool['segments'])[indices], minlength=len(pool['segments'])).tolist() != pool['quotas']:
            raise ProtocolError('Final allocation differs from frozen coarse quota')
        if record['application_cache_hits'] != 0:
            raise ProtocolError('Primary method timing reused application cache')
        if variant in ('A', 'B') and len(rows) != pool['initial_count']:
            raise ProtocolError('No-refinement variant added candidates')
        if variant == 'A' and pool['lambda_value'] != .5:
            raise ProtocolError('Wrong fixed lambda')
        if variant in ('B', 'C'):
            expected_lambda = {'LOCAL': .2, 'GLOBAL': .8, 'MIXED': .5}[pool['scope']['label']]
            if pool['lambda_value'] != expected_lambda:
                raise ProtocolError('Scope/lambda mismatch')
    elif pool['selected_frames'] != select_topk(rows, scores):
        raise ProtocolError('Diagnostic does not use exact global Top-K')
    times = record['timings']
    if any(not math.isfinite(v) or v < -1e-5 for v in times.values()):
        raise ProtocolError('Invalid stage timing')
    if abs(times['end_to_end_seconds'] - times['stage_sum_seconds'] - times['unattributed_seconds']) > 1e-6:
        raise ProtocolError('Stage timings do not reconcile')
    if times['unattributed_seconds'] > max(1., .05 * times['end_to_end_seconds']):
        raise ProtocolError('Excessive unexplained timing overhead')
    return True


def validate_group(group, row, fingerprint, baseline):
    """验收接口：核对同题四组、历史Top-K和C/D候选池关系；只验证执行条件，不以答题正确率决定放行。"""
    for variant in VARIANTS:
        validate_result(group[variant], row, fingerprint)
    pools = {v: group[v]['pool'] for v in VARIANTS}
    if len({group[v]['physical_gpu'] for v in VARIANTS}) != 1:
        raise ProtocolError('Variants for one video used different GPUs')
    for v in ('A', 'B', 'C'):
        n = pools[v]['initial_count']
        if pools[v]['candidates'][:n] != baseline['selection']['candidates']:
            raise ProtocolError('Coarse candidate pool differs from the frozen baseline')
        if not np.allclose(pools[v]['scores'][:n], baseline['selection']['scores'], atol=1e-6, rtol=0):
            raise ProtocolError('BLIP coarse scores differ from frozen baseline')
    if pools['B']['scope']['label'] != pools['C']['scope']['label']:
        raise ProtocolError('Independent deterministic scope classifications disagree')
    if pools['B']['scope']['label'] == 'MIXED' and pools['A']['selected_indices'] != pools['B']['selected_indices']:
        raise ProtocolError('MIXED should reproduce fixed-lambda selection')
    if len(pools['C']['candidates']) == pools['C']['initial_count'] and pools['C']['selected_indices'] != pools['B']['selected_indices']:
        raise ProtocolError('No new candidates but refinement changed selection')
    if pools['C']['candidates'] != pools['D']['candidates'] or pools['C']['scores'] != pools['D']['scores']:
        raise ProtocolError('Diagnostic candidate pool is not exactly the C donor pool')
    if group['D']['donor_result_sha256'] != group['D']['expected_donor_result_sha256']:
        raise ProtocolError('Diagnostic donor identity changed')


def check_recovery(run, rows, fingerprint):
    """恢复接口：校验已完成结果和特征哈希并返回完成集合；对不确定生成或未恢复的已完成分类停止处理，避免重复模型调用。"""
    by_id = {r['question_id']: r for r in rows}
    done = set()
    for v in VARIANTS:
        for p in (Path(run) / 'results' / v).glob('*.json'):
            record = read_json(p)
            if record['question_id'] not in by_id:
                raise ProtocolError('Unexpected question in method results')
            validate_result(record, by_id[record['question_id']], fingerprint)
            feature = Path(run) / record['feature_file']
            if sha256(feature) != record['feature_sha256']:
                raise ProtocolError('Saved feature file changed')
            done.add((record['question_id'], v))
    for p in (Path(run) / 'attempts').glob('*/*.json'):
        attempt = read_json(p)
        key = (attempt['question_id'], attempt['variant'])
        if attempt['status'] in ('running_may_generate', 'uncertain') and key not in done:
            raise ProtocolError(f'Uncertain generation requires inspection: {p}')
        if attempt['status'] == 'failed' and attempt.get('scope_state', {}).get('returned') and key not in done:
            raise ProtocolError(f'Completed classification requires explicit recovery, not repetition: {p}')
    return done
