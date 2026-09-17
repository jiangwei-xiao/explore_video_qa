"""Validation and recovery rules shared by execution and reporting."""
import math
from pathlib import Path
from .common import read_json
from .protocol import parse_answer
from .video import uniform_selection
from .baseline_selection import select_topk, ProtocolError

METHODS = ('uniform', 'topk')


def result_path(run_dir, method, qid):
    return Path(run_dir) / 'results' / method / f'{qid}.json'


def validate_result(record, row, protocol_hash, spec=None):
    """按传入的运行协议校验帧预算；无spec时兼容旧调用及答案协议标记。"""
    if record['status'] != 'completed' or record['protocol_sha256'] != protocol_hash:
        raise ProtocolError('Result status/protocol mismatch')
    if record['question_id'] != row['question_id'] or record['video_id'] != row['video_id']:
        raise ProtocolError('Result identity mismatch')
    if record['question'] != row['question'] or record['options'] != row['options']:
        raise ProtocolError('Original question/options changed')
    method = record['method']
    if method not in METHODS:
        raise ProtocolError('Unexpected method')
    selection, answer = record['selection'], record['answer']
    candidates, selected = selection['candidates'], selection['selected_frames']
    if [r['candidate_index'] for r in candidates] != list(range(len(candidates))):
        raise ProtocolError('Candidate IDs are not contiguous')
    expected = uniform_selection(candidates) if method == 'uniform' else select_topk(candidates, selection['scores'])
    # 2026-09-17 修订：16帧预算为最大上限；候选不足时选帧数 = min(16, 候选数)，视觉token随之按帧数计算
    effective = min(16, len(candidates))
    answer_protocol = answer.get('protocol', 'videoqa-llava-16-v1')
    if spec is not None:
        declared = 'videoqa-llava-16-cap-v1' if 'frame_budget' in spec else 'videoqa-llava-16-v1'
        if answer_protocol != declared:
            raise ProtocolError('Answer protocol differs from frozen run')
    if answer_protocol not in ('videoqa-llava-16-v1', 'videoqa-llava-16-cap-v1'):
        raise ProtocolError('Unknown answer protocol')
    if answer_protocol == 'videoqa-llava-16-v1' and effective != 16:
        raise ProtocolError('Historical fixed-16 protocol requires 16 frames')
    if selected != expected or len(selected) != effective or len({r['source_pts'] for r in selected}) != effective:
        raise ProtocolError('Selection disagrees with the frozen rule')
    if answer['pixel_shape'] != [effective, 3, 384, 384] or answer['visual_tokens'] != effective * 210:
        raise ProtocolError('Visual input protocol mismatch')
    if answer['prefill_tokens'] != answer['text_input_tokens'] - 1 + effective * 210:
        raise ProtocolError('Unexpected multimodal truncation')
    if answer['parsed_answer'] != parse_answer(answer['raw_output']):
        raise ProtocolError('Incorrect answer parsing')
    reference = 'ABCD'[row['answer_index']]
    if record['reference_answer'] != reference or record['correct'] != (answer['parsed_answer'] == reference):
        raise ProtocolError('Incorrect scoring')
    if record['application_cache_hits'] != 0 or record['generation_attempts'] != 1:
        raise ProtocolError('Unexpected cache reuse or generation count')
    times = record['timings']
    if any(not math.isfinite(v) or v < -1e-5 for v in times.values()):
        raise ProtocolError('Non-finite or negative timing')
    if abs(times['stage_sum_seconds'] + times['unattributed_seconds'] - times['end_to_end_seconds']) > 1e-6:
        raise ProtocolError('Timing intervals do not reconcile')
    if times['unattributed_seconds'] > max(1.0, times['end_to_end_seconds'] * 0.05):
        raise ProtocolError('Unexplained timing overhead exceeds 5% or one second')
    if method == 'uniform' and (selection['scores'] or times['blip_forward_seconds'] != 0):
        raise ProtocolError('Uniform unexpectedly ran BLIP')
    if method == 'topk' and not 1 <= selection['blip_question_tokens'] <= 512:
        raise ProtocolError('Invalid BLIP question token length')
    return True


def validate_pair(records, row, protocol_hash, spec=None):
    for method in METHODS:
        validate_result(records[method], row, protocol_hash, spec)
    left, right = records['uniform'], records['topk']
    if left['physical_gpu'] != right['physical_gpu']:
        raise ProtocolError('The paired methods ran on different GPUs')
    if left['selection']['candidates'] != right['selection']['candidates']:
        raise ProtocolError('Paired methods did not have the same candidate pool')
    return True


def check_recovery(run_dir, rows, protocol_hash):
    """Completed results win over an interrupted journal update; uncertain work stops."""
    by_id = {r['question_id']: r for r in rows}
    path = Path(run_dir) / 'protocol.json'
    spec = read_json(path) if path.exists() else None
    completed = set()
    for method in METHODS:
        for path in (Path(run_dir) / 'results' / method).glob('*.json'):
            record = read_json(path)
            if record['question_id'] not in by_id:
                raise ProtocolError('Result is outside the frozen question manifest')
            validate_result(record, by_id[record['question_id']], protocol_hash, spec)
            completed.add((record['question_id'], method))
    for path in (Path(run_dir) / 'attempts').glob('*/*.json'):
        attempt = read_json(path)
        key = (attempt['question_id'], attempt['method'])
        if attempt['status'] in ('running_may_generate', 'uncertain') and key not in completed:
            raise ProtocolError(f'Uncertain interrupted attempt requires inspection: {path}')
    return completed
