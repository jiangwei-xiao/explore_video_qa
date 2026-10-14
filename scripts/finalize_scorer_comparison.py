"""评分器实验收尾：补齐物理成本统计，并实测完成恢复不会产生新调用。"""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import subprocess
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_runtime.baseline_report import distribution
from videoqa_full.state import durable, utc


def generation_identity(root):
    """接口：覆盖两配置实际调用、原始返回及方法结果的字节和文件集合，不加载模型。"""
    records = {}
    # 步骤1：共享引用也属于最终结果；新增或修改任一文件都不能宣称恢复无变化。
    for profile in ('qwen32', 'llava8'):
        for folder in ('calls', 'raw_generations', 'results', 'imported'):
            for path in sorted((root / 'qa' / profile / folder).rglob('*.json')):
                records[str(path.relative_to(root))] = sha256(path)
    return records


def physical_costs(root, profile):
    """接口：只计算物理主调用的成本，共享和历史引用不伪造独立推理耗时。"""
    measures = {}; types = {}; kinds = {}
    # 步骤1：按全部方法结果统计实际题型，但性能仅纳入本轮独立生成。
    for path in sorted((root / 'qa' / profile / 'results').glob('*/*.json')):
        item = read_json(path); method = item['method']; kind = item['timing_kind']
        kinds.setdefault(method, {})[kind] = kinds.setdefault(method, {}).get(kind, 0) + 1
        metric = item['answer']['framework_metric']
        task = metric['task_category']
        cell = types.setdefault(task, {}).setdefault(method, dict(n=0, correct=0))
        cell['n'] += 1; cell['correct'] += int(item['correct'])
        if kind != 'physical_primary':
            continue
        obs = item['answer']['observed']; timing = item['timings']
        values = dict(qa_seconds=obs['qa_seconds'],
                      decode_and_identity_seconds=timing['final_decode_and_identity_seconds'],
                      cached_input_to_answer_seconds=timing['cached_input_to_answer_seconds'])
        # 步骤2：实际后端有记录才汇总，不把别名推测或模型常驻量冒充增量峰值。
        for key in ('generation_seconds', 'peak_allocated_gib', 'peak_reserved_gib',
                    'visual_tokens', 'actual_prefill_tokens'):
            if key in obs and isinstance(obs[key], (int, float)):
                values[key] = obs[key]
        for key, value in values.items():
            measures.setdefault(method, {}).setdefault(key, []).append(value)
    return dict(timing_kind_counts=kinds, official_task_categories=types,
                physical_performance={m: {k: distribution(v) for k, v in fields.items()}
                                      for m, fields in measures.items()})


def preparation_costs(root):
    """接口：粗评分逐题成本与同视频共享解码分开汇总，避免把解码成本重复算三次。"""
    result = {}
    # 步骤1：以正式完成检查点统计真实前向；预热与模型加载另取工作进程记录。
    for stage in ('coarse', 'qwen32', 'llava8'):
        timing = [read_json(p)['timing'] for p in sorted((root / 'checkpoints' / stage).glob('*.json'))]
        keys = sorted({k for t in timing for k, v in t.items() if isinstance(v, (int, float))})
        result[stage] = dict(questions=len(timing),
            measurements={k: dict(**({'total': sum(t[k] for t in timing if k in t)}
                                     if k.endswith(('_seconds', '_frames', '_batches')) else {}),
                                 **distribution([t[k] for t in timing if k in t])) for k in keys},
            model_calls={key: sum(t.get('model_calls', {}).get(key, t.get(key, 0)) for t in timing)
                         for key in ('itm_batches', 'itm_frames', 'visual_batches', 'visual_frames')})
    shared = [read_json(p) for p in sorted((root / 'shared_decode').glob('*.json'))]
    result['shared_video_decode'] = dict(records=len(shared),
        measurements={k: dict(total=sum(t[k] for t in shared), **distribution([t[k] for t in shared]))
                      for k in ('decode_seconds', 'rgb_seconds', 'wall_seconds')})
    result['workers'] = [{k: v for k, v in read_json(p).items()
                          if k in ('stage', 'gpu', 'load_seconds', 'warmup_seconds', 'resident_bytes')}
                         for p in sorted((root / 'workers').glob('*.json'))]
    result['mixed_diagnostic'] = read_json(root / 'mixed_diagnostic/summary.json')
    return result


def task_comparison(root):
    """接口：从已核验的全量四格题号复算S1/S2官方题型，不重复读取大规模像素Token或调用模型。"""
    from videoqa_full.state import rows
    data = rows(); result = {}
    # 步骤1：四格记录已保存逐题答案对错；这里只恢复集合，不附造任何预测。
    for profile, summary in read_json(root / 'comparison.json')['profiles'].items():
        banks = {'S1': {}, 'S2': {}}
        for name, value in summary['paired'].items():
            cells = value['all2700']['cells']
            if name.startswith('S2_minus_S1__'):
                method = name.split('__', 1)[1]
                banks['S2'][method] = set(cells['both_correct'] + cells['left_only'])
                banks['S1'][method] = set(cells['both_correct'] + cells['right_only'])
        uniform = summary['paired']['S2__RD-1.2-exp07_minus_BASE-Uniform']['all2700']['cells']
        for scorer in banks: banks[scorer]['BASE-Uniform'] = set(uniform['both_correct'] + uniform['right_only'])
        # 步骤2：题型来自冻结数据字段，各方法在同一类别分母上计数。
        result[profile] = {s: {str(t): {m: dict(n=sum(r['task_type'] == t for r in data),
            correct=sum(r['question_id'] in correct for r in data if r['task_type'] == t))
            for m, correct in methods.items()} for t in sorted({r['task_type'] for r in data})}
            for s, methods in banks.items()}
    return result


def main():
    """接口：全量完成后调用一次；恢复只运行审计阶段，不触发任何模型生成。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        raise ValueError('Invalid run ID')
    root = ROOT / 'outputs/full_videoqa' / args.run_id
    # 步骤1：完整统计和两个预算验收必须到齐，不能为等待中的运行提前出完成报告。
    if read_json(root / 'comparison.json')['status'] != 'completed':
        raise ValueError('Full comparison incomplete')
    for profile in ('qwen32', 'llava8'):
        expected = read_json(root / 'qa' / profile / 'input_summary.json')['new_calls']
        check = read_json(root / 'qa' / profile / 'completion_audit.json')
        if check['status'] != 'passed' or check['method_results'] != 8100 or check['new_generations'] != expected:
            raise ValueError('Call/results audit differs: ' + profile)
    before = generation_identity(root)
    # 步骤2：真实执行完成恢复入口；不重新执行评分、补查、选择或正式问答。
    subprocess.run(['/App/conda/envs/explore_video_qa_lmms_qwen/bin/python',
                    str(ROOT / 'scripts/resume_scorer_comparison.py'), '--run-id', args.run_id,
                    '--stage', 'audit'], cwd=ROOT, check=True)
    after = generation_identity(root)
    if before != after:
        raise ValueError('Completion recovery changed generation artifacts')
    value = dict(status='passed', new_generations_on_recovery=0,
                 unchanged_generation_artifact_count=len(before), generation_artifacts=after,
                 profiles={p: physical_costs(root, p) for p in ('qwen32', 'llava8')},
                 preparation=preparation_costs(root),
                 official_task_comparison=task_comparison(root),
                 utc=utc())
    durable(root / 'independent_audit/finalization.json', value)
    print(dict(status=value['status'], unchanged_files=len(before), new_generations=0), flush=True)


if __name__ == '__main__':
    main()
