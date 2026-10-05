"""零模型调用的全量控制重放、M16合法池与跨模型共享RGB准备。"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, exclusive, rows, utc
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_budget32.selection import config, OLD_QWEN
from videoqa_methods.retarget_seed_pool import retarget_pool
from videoqa_methods.seed_budget_selection import select_seed_budget

BASE = ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
PARENT = ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
WFS = ROOT/'outputs/analysis/wfs_release_intake_20261004_r1'
WFS_DEV = ROOT/'outputs/diagnostics/wfs_release__qwen32__dev200__20261004_r1'
LLAVA_BASE = ROOT/'outputs/full_videoqa/topk_rd12__llava__lmms_ref_v1__videomme2700__20260923__r01'
LLAVA_UNIFORM = ROOT/'outputs/baselines/base_uniform__llava__lmms_ref_v1__videomme2700__20260923__r01'
DEV = ROOT/'data/manifests/video_mme_development_200_20260920.json'


def frozen(path, value):
    """接口：首次原子保存，恢复逐字段核对，不覆盖冻结资产。"""
    path = Path(path)
    if path.exists():
        if read_json(path) != value: raise ValueError(f'Frozen asset changed: {path}')
    else: durable(path, value)


def prepare_one(job):
    """接口：一题先重放M32控制，再重建M16/K16，输出共享源帧、RGB及独立参考输入。"""
    run, row = job; q = row['question_id']; start = time.perf_counter()
    paths = dict(pool=BASE/'pools'/(q+'.json'), features=BASE/'features'/(q+'.npy'),
                 control=PARENT/'selections/RD-1.2-exp12'/(q+'.json'),
                 reference=WFS/'mapped_selections'/(q+'.json'),
                 checkpoint=BASE/'selection_checkpoints'/(q+'.json'))
    sources = {str(p): sha256(p) for p in paths.values()}
    checkpoint = run/'inputs/checkpoints'/(q+'.json')
    # 步骤1：完成恢复只查完整指纹；部分准备不能作为已通过控制重放。
    if checkpoint.exists():
        record = read_json(checkpoint)
        if record['sources'] != sources: raise ValueError('Input source changed')
        for p, h in record['files'].items():
            if sha256(run/p) != h: raise ValueError('Prepared input changed')
        return record
    pool = read_json(paths['pool']); features = np.load(paths['features'], allow_pickle=False)
    old = read_json(paths['control'])
    source_check = read_json(paths['checkpoint'])
    for name in ['pool','features']:
        if sources[str(paths[name])] != source_check['files'][str(paths[name].relative_to(BASE))]:
            raise ValueError('Original pool or feature identity changed')
    for path, digest in old['sources'].items():
        if sha256(path) != digest: raise ValueError('Parent selection source changed')
    if sources[str(paths['reference'])] != read_json(WFS/'audit.json')['exports'][str(paths['reference'])]:
        raise ValueError('Published reference mapping changed')
    control, _ = select_seed_budget(pool, features, frame_budget=32)
    if control != old['selected_indices']: raise ValueError(f'M32 control mismatch: {q}')
    # 步骤2：按有理数目标筛回合法M16细帧；缺映射暂停，不临时计算新BLIP特征。
    narrow, nf = retarget_pool(pool, features, config(16, 16))
    selected, trace = select_seed_budget(narrow, nf, frame_budget=16)
    frames = [narrow['candidates'][i] for i in selected]
    selection = dict(video=narrow['video'], selected_frames=frames)
    validate_frames(selection, budget=16)
    images, hashes = FrameProvider(selection).decode()
    for image in images: image.close()
    files = {}
    items = {
        'inputs/m16/'+q+'.json': dict(question_id=q, video_id=row['video_id'], method='RD-1.2-exp12',
            selection=selection, rgb_sha256=hashes, sources=sources, selected_indices=selected,
            original_candidate_indices=[narrow['original_candidate_indices'][i] for i in selected],
            configuration='M16/K16', new_model_calls=0),
        'inputs/traces/'+q+'.json': trace,
        'inputs/pools/'+q+'.json': narrow,
    }
    # 步骤3：外部清单逐题精确解码，保持发布源位置；不导入答案至输入层。
    reference = read_json(paths['reference'])
    if reference['source_video_sha256'] != read_json(run/'preflight.json')['video_hashes'][row['video_id']]:
        raise ValueError('Reference video identity mismatch')
    rs = reference['selection']; validate_frames(rs)
    if len(rs['selected_frames']) != 32: raise ValueError('Reference source budget changed')
    images, rh = FrameProvider(rs).decode()
    for image in images: image.close()
    items['inputs/wfs/'+q+'.json'] = dict(question_id=q, video_id=row['video_id'], method='REF-WFS-SB',
        selection=rs, rgb_sha256=rh, sources={str(paths['reference']): sources[str(paths['reference'])]},
        configuration='K32 published source list')
    for rel, value in items.items():
        frozen(run/rel, value); files[rel] = sha256(run/rel)
    record = dict(question_id=q, sources=sources, files=files, m32_control_exact=True,
        m16_frames=len(frames), wfs_frames=32, preparation_seconds=time.perf_counter()-start,
        new_model_calls=0)
    frozen(checkpoint, record); return record


def preflight(run):
    """接口：核对旧环境、全部视频和模型字节，冻结输入来源；本轮GPU前向为0。"""
    from videoqa_budget32.run import verify as verify_qwen
    from videoqa_lmms_baseline.run import verify as verify_llava
    from videoqa_budget32.relevance_full import release_check
    release_check()
    data = rows(); qb = verify_qwen(BASE); lb = verify_llava(LLAVA_UNIFORM)
    if read_json(PARENT/'completion_audit.json')['status'] != 'passed':
        raise ValueError('Parent completion audit incomplete')
    if read_json(WFS_DEV/'completion_audit.json')['status'] != 'passed':
        raise ValueError('Reference development audit incomplete')
    sources = {str(p): sha256(p) for p in [BASE/'protocol.json', PARENT/'protocol.json',
        WFS_DEV/'protocol.json', WFS/'audit.json', DEV, LLAVA_BASE/'protocol.json', LLAVA_UNIFORM/'protocol.json',
        ROOT/'configs/retrieval_density_v1.json']}
    previous = run/'preflight.json'
    if previous.exists():
        saved = read_json(previous)
        if saved['sources'] != sources: raise ValueError('Preflight sources changed')
        return saved
    # 步骤1：视频按900个唯一ID核验，原路径与字节一致，不以文件存在代替内容检查。
    for i, (vid, st) in enumerate(qb['video_stats'].items(), 1):
        if sha256(st['path']) != qb['video_hashes'][vid]: raise ValueError('Video bytes changed')
        if i % 50 == 0:
            durable(run/'preflight_status.json', dict(status='video_hashing', completed=i, total=900, utc=utc()))
    for p, item in {**qb['model_files'], **lb['model_files']}.items():
        if sha256(p) != item['sha256']: raise ValueError('Model bytes changed')
    # 步骤2：历史基线每组2700题闭合，并独立按原解析字母复算；答案仅用于事后比较。
    source_runs = {'qwen16': OLD_QWEN, 'qwen32': BASE, 'llava16': LLAVA_BASE}
    counts = {}
    for label, directory in source_runs.items():
        methods = ('BASE-Uniform', 'BASE-BLIP-TopK', 'RD-1.2')
        for method in methods:
            count = 0
            for row in data:
                if label == 'llava16' and method == 'BASE-Uniform':
                    path = LLAVA_UNIFORM/'results'/(row['question_id']+'.json')
                else: path = directory/'results'/method/(row['question_id']+'.json')
                item = read_json(path)
                answer = item['answer'] if 'answer' in item else item['result']['answer']
                correct = answer['parsed_answer'] == 'ABCD'[row['answer_index']]
                if correct != item['correct'] or item['question'] != row['question'] or item['options'] != row['options']:
                    raise ValueError('Historical identity or scoring mismatch')
                count += correct
            counts[label+'__'+method] = count
    result = dict(status='passed', sources=sources, video_hashes=qb['video_hashes'],
        video_stats=qb['video_stats'], model_files={**qb['model_files'], **lb['model_files']},
        environments=dict(qwen=qb['environment'], llava=lb['environment']),
        baseline_counts=counts, question_count=2700, new_model_calls=0)
    frozen(previous, result); return result


def prepare_inputs(run):
    """接口：有界CPU并行准备2700题；异常停止本阶段，不启动任何正式生成。"""
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    with exclusive(run/'prepare.lock'):
        preflight(run); start = time.perf_counter(); records = []
        # 步骤1：BLAS各2线程，最多8任务同时重放；状态与每题完成检查点独立持久化。
        with ThreadPoolExecutor(max_workers=8) as executor:
            for record in executor.map(prepare_one, [(run, row) for row in rows()]):
                records.append(record)
                durable(run/'input_status.json', dict(status='preparing', completed=len(records), total=2700,
                    elapsed_seconds=time.perf_counter()-start, utc=utc()))
        # 步骤2：全量输入完成后冻结汇总。两后端消费同一m16文件，不产生各自选帧。
        summary = dict(status='passed', questions=2700, m32_control_exact=2700, new_model_calls=0,
            m16_frame_counts={str(n): sum(r['m16_frames']==n for r in records) for n in sorted({r['m16_frames'] for r in records})},
            files={str(run/'inputs/checkpoints'/(r['question_id']+'.json')): sha256(run/'inputs/checkpoints'/(r['question_id']+'.json')) for r in records})
        frozen(run/'input_summary.json', summary)
        durable(run/'input_execution.json', dict(wall_seconds=time.perf_counter()-start, utc=utc()))
        return summary
