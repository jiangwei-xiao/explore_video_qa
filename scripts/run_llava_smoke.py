"""One frozen development question, with a persistent generation-attempt record."""
import argparse
import datetime
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from videoqa_runtime.common import ROOT, DEVELOPMENT_SHA256, offline_environment, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question-id', default='264-2')
    parser.add_argument('--gpu', default='0', help='One physical CUDA device index or UUID')
    parser.add_argument('--output', type=Path, required=True, help='New result path; existing runs are never overwritten')
    args = parser.parse_args()
    if ',' in args.gpu:
        parser.error('Choose exactly one GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    offline_environment()
    manifest_path = ROOT / 'data/manifests/video_mme_development.json'
    if sha256(manifest_path) != DEVELOPMENT_SHA256:
        raise RuntimeError('Frozen development manifest changed')
    rows = [r for r in read_json(manifest_path)['rows'] if r['question_id'] == args.question_id]
    if len(rows) != 1:
        raise ValueError('Question must be in the frozen development set')
    row = rows[0]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write('{}\n')
    record = {'question_id': args.question_id, 'video_id': row['video_id'],
              'purpose': 'runtime_acceptance', 'method': 'uniform', 'status': 'preparing',
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'physical_gpu': args.gpu, 'manifest_sha256': DEVELOPMENT_SHA256,
              'generation_attempts': 0, 'question': row['question'], 'options': row['options'],
              'adapter_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'src/videoqa_runtime').glob('*.py'))}}
    begin = time.perf_counter()
    write_json(output, record)
    try:
        from videoqa_runtime.video import index_video, uniform_selection, decode_selected
        from videoqa_runtime.llava_backend import LlavaBackend
        video_path = (ROOT / 'data' / row['video_relative_path']).resolve()
        if not video_path.is_relative_to((ROOT / 'data/videos').resolve()):
            raise ValueError('Unexpected video path')
        index = index_video(video_path)
        selected = uniform_selection(index['candidates'])
        frames = decode_selected(index, selected)
        record['video_index'] = index
        record['selected_frames'] = selected
        record['decode_total_seconds'] = time.perf_counter() - begin
        record['status'] = 'loading_model'
        write_json(output, record)
        print(f'Decoded {len(frames)} exact source frames from {len(index["candidates"])} candidates.', flush=True)
        backend = LlavaBackend()
        record['model'] = backend.load_report
        write_json(output, record)
        print('Model loaded; starting one real generation.', flush=True)

        def before_generate(inputs):
            record['status'] = 'generation_started'
            record['generation_attempts'] += 1
            record['generation_inputs'] = inputs
            write_json(output, record)

        result = backend.answer(frames, [r['timestamp_seconds'] for r in selected],
                                index['duration_seconds'], row['question'], row['options'], before_generate)
        record['result'] = result
        # The reference answer is used only after inference, never by frame selection or prompting.
        reference = 'ABCD'[row['answer_index']]
        record['reference_answer'] = reference
        record['correct'] = result['parsed_answer'] == reference
        record['status'] = 'completed'
        record['runtime_acceptance_passed'] = True
        record['wall_seconds'] = time.perf_counter() - begin
        write_json(output, record)
        print({'output': str(output), 'raw_output': result['raw_output'],
               'prefill_tokens': result['prefill_tokens'], 'visual_tokens': result['visual_tokens'],
               'generation_seconds': result['generation_seconds'], 'peak_allocated_gib': result['peak_allocated_gib']}, flush=True)
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                      wall_seconds=time.perf_counter() - begin)
        write_json(output, record)
        raise


if __name__ == '__main__':
    main()
