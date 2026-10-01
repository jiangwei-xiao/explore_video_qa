#!/usr/bin/env python3
"""开发200题入口独立对照：后台完成评分、冻结输入，再复用正式Qwen调用账本。"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, utc, rows
from videoqa_audit import query_input_probe as base
from videoqa_audit.option_rank_selection import select_option_ranks
from videoqa_budget32.frames import FrameProvider

METHOD = 'RD-1.2-diag01'
PARENT = ROOT/'outputs/analysis/query_input_pilot9_20261001_r1'
CONTROL = ROOT/'outputs/analysis/option_routes_controls7_20261001_r1'
DATA = ROOT/'data/manifests/video_mme_development_200_20260920.json'
QA_PYTHON = '/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'


def verify(run):
    """接口：恢复核对固定代码、输入、模型和依赖；不接受悄悄变更。"""
    spec = read_json(run/'protocol.json')
    base.verify(PARENT)
    for path, value in spec['sources'].items():
        assert sha256(path) == value, path
    for q, value in spec['inputs'].items():
        assert sha256(run/'inputs'/(q+'.json')) == value
    return spec


def prepare(run, gpus):
    """接口：事前冻结200题、四路查询、缓存来源及200次问答上限。"""
    from transformers import BlipProcessor
    run.mkdir(parents=True, exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():
            return verify(run)
        assert gpus and len(set(gpus)) == len(gpus) and set(gpus) <= set(map(str, range(8)))
        parent = base.verify(PARENT)
        ids = [r['question_id'] for r in read_json(DATA)['rows']]
        assert len(ids) == len(set(ids)) == 200
        data = {r['question_id']: r for r in rows()}
        tokenizer = BlipProcessor.from_pretrained(base.MODEL, local_files_only=True).tokenizer
        sources = {str(DATA): sha256(DATA), str(PARENT/'protocol.json'): sha256(PARENT/'protocol.json')}
        for name in ('scripts/run_option_rank_development.py', 'src/videoqa_audit/option_rank_selection.py'):
            sources[str(ROOT/name)] = sha256(ROOT/name)
            target = run/'code_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        inputs = {}; frames = batches = reused = 0
        old_protocol = read_json(base.SOURCE/'protocol.json')
        # 步骤1：按冻结名单取原题与四选项，评分输入不携带答案或人工事实。
        for q in ids:
            row = data[q]; pp = base.BASE/'pools'/(q+'.json'); fp = base.BASE/'features'/(q+'.npy')
            cp = base.BASE/'selection_checkpoints'/(q+'.json'); check = read_json(cp)
            for p in (pp, fp):
                assert sha256(p) == check['files'][str(p.relative_to(base.BASE))]
            pool = read_json(pp); n = pool['initial_count']; assert n == len(pool['initial_scores'])
            assert sha256(pool['video']['path']) == old_protocol['video_hashes'][row['video_id']]
            queries = [row['question']+'\n'+option for option in row['options']]
            assert len(queries) == 4
            lengths = [len(tokenizer(x, truncation=False)['input_ids']) for x in queries]
            assert max(lengths) <= parent['position_limit']
            cache = None
            if (CONTROL/'results'/(q+'.json')).exists():
                previous = read_json(CONTROL/'inputs'/(q+'.json'))
                assert previous['queries'] == queries and previous['pool_path'] == str(pp)
                cache = str(CONTROL/'results'/(q+'.json')); reused += 1
                for p in (Path(cache), CONTROL/'protocol.json', CONTROL/'inputs'/(q+'.json'), PARENT/'results'/(q+'.json')):
                    sources[str(p)] = sha256(p)
            else:
                frames += 4*n; batches += 4*((n+15)//16)
            for p in (pp, fp, cp, Path(pool['video']['path']), base.FULL/'imported/results/RD-1.2-exp07'/(q+'.json')):
                sources[str(p)] = sha256(p)
            item = dict(question_id=q, video_id=row['video_id'], queries=queries, query_tokens=lengths,
                        pool_path=str(pp), feature_path=str(fp), initial_count=n, cache=cache)
            durable(run/'inputs'/(q+'.json'), item); inputs[q] = sha256(run/'inputs'/(q+'.json'))
        # 步骤2：只冻结一种融合，不根据九题或后续回答选择多个参数版本。
        spec = dict(method=METHOD, question_ids=ids, allowed_gpus=gpus, inputs=inputs, sources=sources,
                    normal_frame_text_scores=frames, normal_batch_calls=batches, cached_questions=reused,
                    max_qa_calls=200, budget=32, configuration='four option routes; minimum within-route rank; direct coarse Top32',
                    no_refinement=True, no_classifier=True, no_query_generation=True, utc=utc())
        durable(run/'protocol.json', spec)
        return spec


def score_question(run, q, scorer, gpu):
    """接口：有界解码、四路完整评分；纯视觉身份不符或未知调用状态立即暂停。"""
    item = read_json(run/'inputs'/(q+'.json')); dest = run/'results'/(q+'.json')
    if dest.exists():
        assert read_json(dest)['protocol_sha256'] == sha256(run/'protocol.json')
        return
    pool = read_json(item['pool_path']); coarse = pool['candidates'][:item['initial_count']]
    if item['cache']:
        cached = read_json(item['cache']); assert cached['status'] == 'completed'
        scores = cached['scores']; rgb = read_json(PARENT/'results'/(q+'.json'))['rgb_sha256']
        error = cached['max_feature_error']
    else:
        frozen = np.load(item['feature_path'], mmap_mode='r', allow_pickle=False)[:len(coarse)]
        encoded = [scorer.tokenize(x) for x in item['queries']]
        scores = [[] for _ in range(4)]; rgb = []; batch = []; seen = 0; error = 0.
        def flush():
            """步骤1同批四文本，步骤2核对纯视觉，步骤3释放RGB并持久保存进度。"""
            nonlocal error
            if (run/'STOP.json').exists():
                raise RuntimeError('Peer failure')
            begin = seen-len(batch)
            hashes = [hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in batch]
            rgb.extend(hashes)
            for k in range(4):
                values, vectors = base.score_once(run, q, f'option_{k}', begin, scorer, encoded[k], batch, gpu, hashes)
                error = max(error, float(np.max(np.abs(vectors-frozen[begin:seen]))))
                assert error <= 1e-6, (q, begin, error)
                scores[k].extend(values)
            for im in batch:
                im.close()
            batch.clear()
            durable(run/'workers'/(gpu+'.json'), dict(question_id=q, completed_frames=seen, total=len(coarse), pid=os.getpid(), utc=utc()))
        for record, frame in base.iter_candidate_frames(pool['video']['path'], {}):
            for key in ('candidate_index', 'source_pts', 'source_frame_index', 'requested_seconds'):
                assert record[key] == coarse[seen][key], (q, key)
            seen += 1; batch.append(frame.to_image())
            if len(batch) == 16:
                flush()
        if batch:
            flush()
        assert seen == len(coarse)
    selected, trace = select_option_ranks(coarse, scores)
    selection = dict(video=pool['video'], selected_frames=[coarse[i] for i in selected])
    # 步骤3：独立精确取最终帧，要求与评分时像素逐帧相同；不把候选编号当源编号。
    images, hashes = FrameProvider(selection).decode()
    for image in images:
        image.close()
    assert hashes == [rgb[i] for i in selected]
    result = dict(status='completed', question_id=q, protocol_sha256=sha256(run/'protocol.json'),
                  scores=scores, selected_indices=selected, trace=trace, rgb_sha256=rgb,
                  max_feature_error=error, cached_from=item['cache'], utc=utc())
    durable(dest, result)


def worker(run, gpu, questions):
    """接口：一卡常驻BLIP；失败写STOP，其他进程完成当前批后停止，不重试。"""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu; offline_environment()
    active = None
    try:
        import torch
        from PIL import Image
        from videoqa_methods.model_ops import FeatureScorer
        torch.set_num_threads(2); torch.set_num_interop_threads(1); base.configure_visual_precision(torch)
        scorer = FeatureScorer(str(base.MODEL))
        for k in range(2):
            p = run/'synthetic'/f'{gpu}_{k}.json'
            if p.exists():
                assert read_json(p)['status'] == 'completed'; continue
            images = [Image.new('RGB', (1280,720), 'gray') for _ in range(16)]
            durable(p, dict(status='started', utc=utc()))
            scorer.score_batch_features(scorer.tokenize('Which synthetic color is visible?'), images)
            for im in images:
                im.close()
            durable(p, dict(status='completed', frames=16, utc=utc()))
        for q in questions:
            active = q
            if (run/'STOP.json').exists():
                raise RuntimeError('Peer failure')
            score_question(run, q, scorer, gpu)
    except BaseException as exc:
        issue = dict(question_id=active, gpu=gpu, error=repr(exc), traceback=traceback.format_exc(), utc=utc())
        durable(run/'failures'/f'{gpu}_{time.time_ns()}.json', issue)
        durable(run/'STOP.json', issue)
        raise


def export_inputs(run, spec):
    """接口：全200题评分闭合后冻结最终帧，供已验收200题执行器使用。"""
    exports = {}; inputs = run/'qa_inputs'
    for q in spec['question_ids']:
        item = read_json(run/'inputs'/(q+'.json')); rp = run/'results'/(q+'.json'); result = read_json(rp)
        assert result['status'] == 'completed' and result['protocol_sha256'] == sha256(run/'protocol.json')
        pool = read_json(item['pool_path']); coarse = pool['candidates'][:item['initial_count']]
        selected, trace = select_option_ranks(coarse, result['scores'])
        assert selected == result['selected_indices'] and trace == result['trace']
        source_paths = [run/'protocol.json', rp, Path(item['pool_path']), ROOT/'src/videoqa_audit/option_rank_selection.py']
        selection = dict(video=pool['video'], selected_frames=[coarse[i] for i in selected])
        value = dict(question_id=q, video_id=item['video_id'], method=METHOD, selection=selection,
                     rgb_sha256=[result['rgb_sha256'][i] for i in selected],
                     sources={str(p):sha256(p) for p in source_paths})
        out = inputs/(q+'.json')
        if out.exists():
            assert read_json(out) == value
        else:
            durable(out, value)
        exports[q] = sha256(out)
    calls = list((run/'calls').glob('*/*/*.json'))
    assert len(calls) == spec['normal_batch_calls']
    assert all(read_json(p)['status'] == 'completed' for p in calls)
    assert sum(read_json(p)['identity']['frames'] for p in calls) == spec['normal_frame_text_scores']
    manifest = dict(status='completed', question_count=200, frame_budget=32, question_ids=spec['question_ids'],
                    dataset_manifest=str(DATA), manifest_sha256=sha256(DATA), parent_run=str(base.BASE),
                    parent_protocol_sha256=sha256(base.BASE/'protocol.json'), method=METHOD,
                    configuration=spec['configuration'], diagnostic=True, exports=exports)
    path = inputs/'selection_manifest.json'
    if path.exists():
        assert read_json(path) == manifest
    else:
        durable(path, manifest)
    return inputs


def execute(run, qa):
    """接口：后台先完成全部选择再问答；恢复沿用评分GPU，未知状态不能自动重启。"""
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(run/'coordinator.lock'):
        spec = verify(run); start = time.perf_counter()
        assert not (run/'STOP.json').exists(), 'Inspect failure before recovery'
        remaining = [q for q in spec['question_ids'] if not (run/'results'/(q+'.json')).exists()]
        if remaining:
            assignment_path = run/'assignment.json'
            # 步骤1：已有分配固定物理GPU；新启动只用实际空闲卡，不抢占或动态扩卡。
            while True:
                try:
                    free, _ = available_gpus(spec['allowed_gpus'])
                except RuntimeError:
                    free = []
                if assignment_path.exists():
                    assignment = read_json(assignment_path)
                    needed = [g for g, qs in assignment.items() if set(qs)&set(remaining)]
                    if set(needed) <= set(free):
                        break
                elif free:
                    assignment = {g:[] for g in free}; loads = {g:0 for g in free}
                    for q in sorted(remaining, key=lambda q: -read_json(run/'inputs'/(q+'.json'))['initial_count']):
                        g = min(free, key=lambda g: (loads[g],g)); assignment[g].append(q)
                        loads[g] += read_json(run/'inputs'/(q+'.json'))['initial_count']
                    durable(assignment_path, assignment); needed = free; break
                durable(run/'status.json', dict(status='waiting_resources', pid=os.getpid(), utc=utc())); time.sleep(15)
            children = []; logs = []
            for g in needed:
                qs = [q for q in assignment[g] if q in remaining]
                (run/'logs').mkdir(exist_ok=True); log = (run/'logs'/f'{g}.log').open('a'); logs.append(log)
                children.append(subprocess.Popen([base.PYTHON, __file__, '--run-id', run.name, '--stage', 'worker',
                    '--gpu', g, '--questions', *qs], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
            while any(p.poll() is None for p in children):
                durable(run/'status.json', dict(status='scoring', completed=len(list((run/'results').glob('*.json'))),
                    total=200, pid=os.getpid(), sid=os.getsid(0), seconds=time.perf_counter()-start, utc=utc()))
                time.sleep(5)
            for log in logs:
                log.close()
            assert all(p.returncode == 0 for p in children), 'Worker failed; no retry'
        # 步骤2：重新核对身份与完整账本；每题输入先冻结，问答不参与选帧。
        verify(run); inputs = export_inputs(run, spec)
        durable(run/'selection_complete.json', dict(status='completed', new_frame_text_scores=spec['normal_frame_text_scores'],
            cached_questions=spec['cached_questions'], seconds=time.perf_counter()-start, utc=utc()))
        if qa:
            # 步骤3：Qwen使用独立环境和原执行器，最多200次；后台自行先行验收和收尾。
            entry = str(ROOT/'scripts/run_frozen_development.py')
            subprocess.run([QA_PYTHON, entry, '--run-id', run.name, '--stage', 'prepare', '--input-dir', str(inputs),
                            '--gpus', *spec['allowed_gpus'], '--resume'], check=True)
            durable(run/'status.json', dict(status='qa', pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            subprocess.run([QA_PYTHON, entry, '--run-id', run.name, '--stage', 'qa', '--resume'], check=True)
        durable(run/'status.json', dict(status='completed' if qa else 'selection_completed', pid=os.getpid(),
            sid=os.getsid(0), seconds=time.perf_counter()-start, utc=utc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=['prepare','run','worker'], required=True)
    parser.add_argument('--gpus', nargs='+', default=list(map(str,range(8))))
    parser.add_argument('--gpu'); parser.add_argument('--questions', nargs='+')
    parser.add_argument('--qa', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id)
    run = ROOT/'outputs/analysis'/args.run_id
    if args.stage == 'prepare':
        prepare(run, args.gpus)
    elif args.stage == 'worker':
        worker(run, args.gpu, args.questions)
    else:
        if (run/'calls').exists() and not args.resume:
            raise RuntimeError('Explicit resume required')
        execute(run, args.qa)
