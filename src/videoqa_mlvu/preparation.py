"""MLVU全片BLIP-1评分与冻结RD选择；模型前向、共享解码和恢复分别记账。"""
from pathlib import Path
from fractions import Fraction
import os
import time
import shutil
import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, save_array, utc
from videoqa_mlvu.data import rows, METHODS, PROFILES, PYTHONS
from videoqa_methods.budget_contract import budget_config, select_exp07
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.density_selection import build_regions
from videoqa_runtime.baseline_selection import iter_candidate_frames, select_topk
from videoqa_budget32.frames import native_frames, validate_frames

BLIP = Path('/home/models/Salesforce/blip-itm-base-coco')
_SCORER = None


def preflight(root):
    """接口：数据、算法、三模型和环境在首次GPU计算前冻结；恢复不接受指纹漂移。"""
    from videoqa_lmms_bridge.common import framework_identity, LMMS
    from videoqa_full.prepare import environment
    root = Path(root)
    if (root / 'protocol.json').exists():
        verify(root); return read_json(root / 'protocol.json')
    if not (root / 'indices_complete.json').exists() or not (root / 'controls_summary.json').exists():
        raise ValueError('Complete full indices and 10800 controls first')
    if read_json(root / 'controls_summary.json')['selection_replays'] != 10800: raise ValueError('Controls incomplete')
    if shutil.disk_usage(root).free < 100 * 2**30: raise ValueError('Insufficient 100 GiB preparation reserve')
    data = rows(root); manifest = read_json(root / 'manifest.json')
    # 步骤1：直接核验既有官方权重身份，不下载、不写共享模型目录。
    inventory = read_json(ROOT / 'configs/local_model_inventory.json'); sources = {}
    for model in inventory['models'].values():
        for name, descriptor in model['files'].items():
            path = Path(model['path']) / name
            if sha256(path) != descriptor['sha256']: raise ValueError('Model identity changed: ' + str(path))
            sources[str(path)] = descriptor['sha256']
    for path, digest in manifest['video_hashes'].items():
        stat = Path(path).stat()
        sources[path] = digest
    # 步骤2：算法字节与新适配层都纳入快照；MLVU任务来自固定干净框架副本。
    code = sorted(set([*ROOT.glob('src/videoqa_mlvu/*.py'), ROOT / 'scripts/run_mlvu_full.py',
        *ROOT.glob('src/videoqa_runtime/*.py'), ROOT / 'src/videoqa_full/state.py', ROOT / 'src/videoqa_budget32/frames.py',
        ROOT / 'src/videoqa_full/qwen_fa2_backend.py',
        *ROOT.glob('src/videoqa_methods/*.py'), ROOT / 'configs/retrieval_density_v1.json',
        ROOT / 'src/videoqa_generalization/llava_observe.py', ROOT / 'src/videoqa_budget32/observe.py',
        ROOT / 'src/videoqa_lmms_bridge/llava_adapter.py', ROOT / 'src/videoqa_budget32/qwen_adapter.py']))
    fingerprints = {str(p.relative_to(ROOT)): sha256(p) for p in code}
    for relative, digest in fingerprints.items():
        dest = root / 'code_snapshot' / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    task_files = {str(p): sha256(p) for p in [LMMS / 'lmms_eval/tasks/mlvu/utils.py', LMMS / 'lmms_eval/tasks/mlvu/mlvu_dev.yaml']}
    tokens = []
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BLIP, local_files_only=True)
    for r in data:
        n = len(tokenizer(r['question'], truncation=False)['input_ids'])
        if n > 512: raise ValueError('BLIP text would truncate: ' + r['question_id'])
        tokens.append(n)
    framework = framework_identity()
    framework.pop('task_reference_commit', None); framework.pop('task_sha256', None)
    framework['task_name'] = 'mlvu_dev'; framework['task_utils_sha256'] = task_files[str(LMMS / 'lmms_eval/tasks/mlvu/utils.py')]
    spec = dict(dataset='MLVU-Dev-MCQ', questions=2174, methods=list(METHODS), profiles=PROFILES,
        scorer_id='S1-BLIP-1-ITM', official_revision=manifest['official_revision'],
        manifest_sha256=sha256(root / 'manifest.json'), sources=sources,
        source_stats={p: dict(size=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns) for p in sources},
        framework=framework, task_files=task_files, code=fingerprints,
        controls_summary_sha256=sha256(root / 'controls_summary.json'),
        environments={k: environment(v) for k, v in PYTHONS.items()},
        text_token_min=min(tokens), text_token_max=max(tokens), generation=dict(max_new_tokens=16, do_sample=False, seed=2027),
        gpu_policy='Idle physical GPUs0-7 only; no preemption or stage expansion', utc=utc())
    durable(root / 'protocol.json', spec); verify(root); return spec


def verify(root, full=False):
    """接口：每阶段检查代码/任务及源文件状态；完整字节核验用于最后验收。"""
    root = Path(root); spec = read_json(root / 'protocol.json')
    from videoqa_lmms_bridge.common import framework_identity
    current = framework_identity()
    if any(current[k] != spec['framework'][k] for k in ('commit','model_sha256','qwen_model_sha256','instance_sha256')):
        raise ValueError('Frozen framework model interface changed')
    if sha256(root / 'manifest.json') != spec['manifest_sha256']: raise ValueError('Manifest fingerprint changed')
    if sha256(root / 'controls_summary.json') != spec['controls_summary_sha256']: raise ValueError('Control replay changed')
    for p, h in spec['code'].items():
        if sha256(ROOT / p) != h: raise ValueError('Frozen run code changed: ' + p)
    for p, h in spec['task_files'].items():
        if sha256(p) != h: raise ValueError('MLVU task changed')
    for p, h in spec['sources'].items():
        st = Path(p).stat(); expected = spec['source_stats'][p]
        if st.st_size != expected['size'] or st.st_mtime_ns != expected['mtime_ns']: raise ValueError('Source stat changed')
        if full and sha256(p) != h: raise ValueError('Source byte fingerprint changed')
    return spec


def completed(root, stage, q):
    """接口：准备完成检查点需绑定全部资产，半份特征不能进入正式问答。"""
    root = Path(root); path = root / 'checkpoints' / stage / (q + '.json')
    if not path.exists(): return False
    c = read_json(path)
    if c['protocol_sha256'] != sha256(root / 'protocol.json'): raise ValueError('Preparation protocol changed')
    for p, h in c['files'].items():
        if sha256(root / p) != h: raise ValueError('Completed preparation changed')
    return True


def checkpoint(root, stage, q, files, timing):
    """接口：绑定本题全部完成资产与父协议，耗时不是独立原片E2E。"""
    root = Path(root)
    durable(root / 'checkpoints' / stage / (q + '.json'), dict(protocol_sha256=sha256(root / 'protocol.json'),
        files={str(p.relative_to(root)): sha256(p) for p in files}, timing=timing, utc=utc()))


def initialize_scorer(root, gpu, stage):
    """接口：BLIP-1独立模型阶段；合成验收、加载和真实题成本严格分开。"""
    global _SCORER
    offline_environment()
    import torch
    from videoqa_methods.density_pipeline import DensityScorer
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(2027)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    begin = time.perf_counter(); _SCORER = DensityScorer(BLIP); loading = time.perf_counter()-begin
    begin = time.perf_counter(); acceptance = _SCORER.synthetic_acceptance()
    durable(Path(root) / 'workers' / (stage+'_'+gpu+'_'+str(os.getpid())+'.json'),
        dict(status='passed', stage=stage, gpu=gpu, load_seconds=loading, warmup_seconds=time.perf_counter()-begin,
             acceptance=acceptance, resident_bytes=torch.cuda.memory_allocated(), utc=utc()))


def coarse_video(root, video_rows):
    """接口：同内容视频一次有界RGB解码，每题原问题独立官方ITM＋视觉CLS前向。"""
    root = Path(root); pending = [r for r in video_rows if not completed(root, 'coarse', r['question_id'])]
    if not pending: return
    if len({r['video_id'] for r in pending}) != 1: raise ValueError('Mixed source videos')
    vid = pending[0]['video_id']; index = read_json(root / 'indices' / (vid + '.json'))
    pts = np.load(index['pts_path'], allow_pickle=False)
    encoded = {r['question_id']: _SCORER.tokenize(r['question']) for r in pending}
    scores = {q: [] for q in encoded}; features = {q: [] for q in encoded}
    costs = {q: dict(preprocess_seconds=0., forward_seconds=0., itm_frames=0, itm_batches=0) for q in encoded}
    frames = []; images = []; metadata = {}; started = time.perf_counter(); decode = rgb = 0.
    # 步骤1：候选映射用已完整解码的源索引复核，不按平均帧率或截短视频近似。
    iterator = iter_candidate_frames(index['video']['path'], metadata)
    def flush():
        try:
            for q in encoded:
                values, f, pre, fw = _SCORER.score_batch_features(encoded[q], images)
                scores[q].extend(values); features[q].append(f)
                costs[q]['preprocess_seconds'] += pre; costs[q]['forward_seconds'] += fw
                costs[q]['itm_frames'] += len(images); costs[q]['itm_batches'] += 1
            durable(root / 'worker_progress' / (str(os.getpid())+'.json'),
                dict(stage='coarse', video_id=vid, candidate_frames=len(frames), questions=len(pending), utc=utc()))
        finally:
            for im in images: im.close()
            images.clear()
    while True:
        t = time.perf_counter()
        try: frame, source = next(iterator)
        except StopIteration: decode += time.perf_counter()-t; break
        decode += time.perf_counter()-t
        target = Fraction(str(frame['requested_seconds'])) / Fraction(index['video']['time_base']) + index['video']['start_pts']
        tick = (target.numerator + target.denominator-1)//target.denominator
        expected = int(np.searchsorted(pts, tick))
        if frame['source_frame_index'] != expected or frame['source_pts'] != int(pts[expected]):
            raise ValueError('Coarse source index/PTS does not match full index')
        frames.append(frame); t = time.perf_counter(); images.append(source.to_image()); rgb += time.perf_counter()-t
        if len(images) == 16: flush()
    if images: flush()
    metadata['duration_fraction'] = index['video']['duration_fraction']
    if metadata != index['video']: raise ValueError('Full index and coarse video clock differ')
    # 步骤2：每题独立记录分数与特征；均匀源帧仅依赖完整编号域，绝不承担评分计算。
    durable(root / 'shared_decode' / (vid + '_' + str(os.getpid()) + '.json'), dict(video_id=vid, question_ids=list(encoded),
        candidate_count=len(frames), decode_seconds=decode, rgb_seconds=rgb, wall_seconds=time.perf_counter()-started, utc=utc()))
    for q in encoded:
        pp = root / 'coarse/pools' / (q+'.json'); fp = root / 'coarse/features' / (q+'.npy')
        durable(pp, dict(video=metadata, candidates=frames, initial_count=len(frames), initial_scores=scores[q],
            blip_question_tokens=int(encoded[q]['input_ids'].shape[1]), scorer_id='S1-BLIP-1-ITM',
            feature_backbone='Salesforce/blip-itm-base-coco', feature_type='pure_visual_CLS_L2'))
        save_array(fp, np.concatenate(features[q], axis=0)); checkpoint(root, 'coarse', q, [pp, fp], costs[q])


def refine_one(root, row):
    """接口：两预算各自重建原补查池；exp07/12共池共前缀，不借用其他探索版本。"""
    from PIL import Image
    from videoqa_methods.density_pipeline import decode_regions
    root = Path(root); q = row['question_id']; coarse = read_json(root / 'coarse/pools' / (q+'.json'))
    original_features = np.load(root / 'coarse/features' / (q+'.npy'), allow_pickle=False)
    for profile, k in PROFILES.items():
        if completed(root, profile, q): continue
        begin = time.perf_counter(); cfg = budget_config(k)
        plan = build_regions(coarse['candidates'], coarse['initial_scores'], coarse['video'], cfg)
        directory = root / 'temporary_rgb' / profile / q
        added, assets, refinement, timing = decode_regions(coarse['video'], coarse['candidates'], plan, cfg, directory)
        source_record = read_json(root / 'indices' / (row['video_id']+'.json'))
        source_pts = np.load(source_record['pts_path'], mmap_mode='r', allow_pickle=False)
        # 用全片int64索引独立确认补查源编号及第一张不早于网格目标的源帧。
        for frame in added:
            target = Fraction(str(frame['requested_seconds'])) / Fraction(coarse['video']['time_base']) + coarse['video']['start_pts']
            tick = (target.numerator+target.denominator-1)//target.denominator
            expected = int(np.searchsorted(source_pts, tick))
            if expected != frame['source_frame_index'] or int(source_pts[expected]) != frame['source_pts']:
                raise ValueError('Refinement source identity does not match full PTS index')
        blocks = [original_features]; pre = fw = 0.
        # 步骤1：补查仅纯视觉编码，保持官方batch16；暂存PNG不是永久实验输入。
        for offset in range(0, len(assets), 16):
            images = []
            for asset in assets[offset:offset+16]:
                with Image.open(asset['path']) as im: images.append(im.convert('RGB'))
            try: f, p, w = _SCORER.visual_batch(images)
            finally:
                for im in images: im.close()
            blocks.append(f); pre += p; fw += w
        features = np.concatenate(blocks, axis=0)
        pool = dict(coarse, candidates=coarse['candidates']+added, plan=plan, refinement=refinement,
                    new_candidate_count=len(added), new_frames_have_itm_scores=False)
        a, trace = select_exp07(pool, features, k); b, weighted = select_relevance_coverage(pool, features, trace)
        index = read_json(root / 'indices' / (row['video_id']+'.json'))
        choices = {METHODS[0]: native_frames(np.load(index['pts_path'], allow_pickle=False), pool['video'], k),
            METHODS[1]: select_topk(coarse['candidates'], coarse['initial_scores'], k),
            METHODS[2]: [pool['candidates'][i] for i in a], METHODS[3]: [pool['candidates'][i] for i in b]}
        pp = root / 'pools' / profile / (q+'.json'); fp = root / 'features' / profile / (q+'.npy')
        tp = root / 'traces' / profile / (q+'.json'); durable(pp, pool); save_array(fp, features)
        durable(tp, dict(exp07=trace, exp12=weighted)); files = [pp, fp, tp]
        # 步骤2：保存完整轨迹和四方法选择，实际预算闭合后才发布完成检查点。
        for method, frames in choices.items():
            selection = dict(video=pool['video'], selected_frames=frames); validate_frames(selection, k)
            dest = root / 'selections' / profile / method / (q+'.json')
            durable(dest, dict(question_id=q, method=method, frame_budget=k, selection=selection)); files.append(dest)
        timing.update(fine_visual_preprocess_seconds=pre, fine_visual_forward_seconds=fw,
            fine_visual_frames=len(assets), fine_itm_frames=0, wall_seconds=time.perf_counter()-begin)
        checkpoint(root, profile, q, files, timing)
        # 步骤3：只有本题资产全部发布后释放明确创建的暂存文件；失败现场不删除。
        for asset in assets:
            p = Path(asset['path'])
            if p.parent != directory or sha256(p) != asset['sha256']: raise ValueError('Unexpected transient file')
            p.unlink()
