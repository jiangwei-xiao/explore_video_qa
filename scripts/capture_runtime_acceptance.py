"""Persist environment locks and a compact, traceable runtime acceptance summary."""
import importlib.metadata
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from videoqa_runtime.common import ROOT, read_json, sha256, write_json


def main():
    result_path = ROOT / 'outputs/acceptance/llava_264-2.json'
    result = read_json(result_path)
    if result['status'] != 'completed' or result['generation_attempts'] != 1:
        raise RuntimeError('Expected a completed, single-attempt real-video acceptance')
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    write_json(ROOT / 'configs/runtime_environment.json', {
        'python': sys.version, 'executable': sys.executable,
        'prefix': sys.prefix, 'environment_type': 'conda', 'packages': versions,
    })
    frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--exclude-editable'], text=True)
    (ROOT / 'requirements-runtime.lock.txt').write_text(frozen)
    conda = Path('/App/conda/bin/conda')
    explicit = subprocess.check_output([str(conda), 'list', '--prefix', sys.prefix, '--explicit'], text=True)
    (ROOT / 'conda-linux-64.lock.txt').write_text(explicit)
    model = result['model']
    answer = result['result']
    summary = {
        'status': 'real_video_runtime_acceptance_passed',
        'completed_qa_calls': 1, 'full_baseline_run_completed': False,
        'result_path': str(result_path.relative_to(ROOT)), 'result_sha256': sha256(result_path),
        'question_id': result['question_id'], 'source_commit': model['source_commit'],
        'model_inventory_sha256': model['model_inventory_sha256'],
        'manifest_sha256': result['manifest_sha256'], 'protocol': answer['protocol'],
        'environment': {'name': 'explore_video_qa', 'prefix': sys.prefix, 'python': sys.version,
                        'torch': model['torch'], 'transformers': versions['transformers']},
        'pixel_shape': answer['pixel_shape'], 'prefill_tokens': answer['prefill_tokens'],
        'visual_tokens': answer['visual_tokens'], 'raw_output': answer['raw_output'],
        'parsed_answer': answer['parsed_answer'], 'correct': result['correct'],
        'generation_seconds': answer['generation_seconds'],
        'peak_allocated_gib': answer['peak_allocated_gib'],
        'checked_logit_steps': answer['checked_logit_steps'],
        'loading_warning_count': sum(model['loading_warnings'].values()),
        'loading_warning_assessment': 'Nested SigLIP meta-copy warnings retained in full record. '
            'No final meta parameters; five deterministic outer-checkpoint tensor slices match exactly; '
            'real-video generation has finite next-token logits. Not an exhaustive in-memory tensor comparison.',
        'checkpoint_tensor_checks': model['checkpoint_tensor_checks'],
        'limits': ['One development question, not an accuracy estimate',
                   'BLIP runtime and full baseline scheduler are not validated here',
                   'Local model SHA256 frozen; upstream LFS identity not compared'],
    }
    write_json(ROOT / 'configs/model_runtime_validation.json', summary)
    print('Conda/pip locks, environment inventory, and acceptance summary saved.')


if __name__ == '__main__':
    main()
