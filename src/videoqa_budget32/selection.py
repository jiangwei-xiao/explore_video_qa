"""32帧选择准备：冻结粗资产复用，独立补查，无真实问答。"""
import os
import time
from pathlib import Path
import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, save_array, rows, exclusive, utc
from videoqa_methods.density_selection import build_regions, refinement_targets
from videoqa_audit.density_reward_decay import select_density_decay
from videoqa_audit.local_distance_scale import select_local_distance
from .frames import native_frames, validate_frames, FrameProvider

SOURCE = ROOT/'outputs/full_videoqa/rd_1_2__videomme2700__crossmodel_20260922__r01'
INDEX = ROOT/'outputs/analysis/lmms_bridge_videomme_20260923_r1'
OLD_QWEN = ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__20260923__r01'
METHODS = ('BASE-Uniform', 'BASE-BLIP-TopK', 'RD-1.2')


def config(budget=32, seeds=32):
    """接口：只覆盖本轮M/K，其他冻结参数保持原样。"""
    result = read_json(ROOT/'configs/retrieval_density_v1.json')
    result.update(seed_budget=seeds, frame_budget=budget)
    return result


def two_pass(pool, features, cfg):
    """接口：直接调用冻结RD-1.2两遍选择，不经过RD-1.0选择器。"""
    # 步骤1：递减密度奖励确定区域配额。
    _, allocation = select_density_decay(pool['candidates'], features, pool['plan'], pool['video'], cfg)
    # 步骤2：固定配额，以L×sqrt(u)选出最终源帧。
    selected, trace = select_local_distance(pool['candidates'], features, pool['plan'], pool['video'], cfg,
                                          allocation['region_budgets'])
    return selected, allocation, trace


def load_old(qid):
    """接口：核对检查点绑定的池与特征字节，禁止混用旧细帧前缀。"""
    check_path = SOURCE/'llava/checkpoints'/(qid+'.json'); check = read_json(check_path)
    pp, fp = SOURCE/check['pool_file'], SOURCE/check['feature_file']
    assert sha256(pp) == check['pool_sha256'] and sha256(fp) == check['feature_sha256']
    pool, features = read_json(pp), np.load(fp, allow_pickle=False)
    assert len(features) == len(pool['candidates'])
    assert len(pool['initial_scores']) == pool['initial_count']
    return pool, features, {str(p):sha256(p) for p in (check_path, pp, fp)}


def precheck_question(job):
    """单题CPU接口：旧16重放、固定旧池32诊断、正式32窗口规划。"""
    run, row = Path(job[0]), job[1]; qid = row['question_id']
    dest = run/'precheck'/(qid+'.json')
    if dest.exists():
        item = read_json(dest)
        for p, h in item['sources'].items(): assert sha256(p) == h
        assert sha256(run/item['diagnostic_file']) == item['diagnostic_sha256']
        return item
    # 步骤1：完整旧池重放，选择变化必须暂停而非重答。
    started = time.perf_counter(); old, features, sources = load_old(qid)
    selected, _, _ = two_pass(old, features, config(16,16))
    assert selected == old['selected_indices'], ('Old replay mismatch', qid)
    selected, allocation, trace = two_pass(old, features, config(32,16))
    diagnostic = dict(question_id=qid, configuration='M16/K32 fixed old pool', new_model_calls=0,
        selected_indices=selected, selected_frames=[old['candidates'][i] for i in selected],
        budget_trace=allocation, selection_trace=trace, source_hashes=sources)
    dp = run/'diagnostics'/(qid+'.json'); durable(dp, diagnostic)
    # 步骤2：只从初始粗前缀构建M32窗口；人工事实不参与。
    coarse = old['candidates'][:old['initial_count']]
    plan = build_regions(coarse, old['initial_scores'], old['video'], config())
    targets = sum(len(refinement_targets(r, old['video'], config())) for r in plan['regions'] if r['refine'])
    index_path = INDEX/'uniform_records'/(row['video_id']+'.json'); index = read_json(index_path)
    assert sha256(INDEX/index['pts_file']) == index['pts_sha256']
    pts = np.load(INDEX/index['pts_file'], allow_pickle=False)
    for f in coarse: assert int(pts[f['source_frame_index']]) == f['source_pts']
    sources[str(index_path)] = sha256(index_path); sources[str(INDEX/index['pts_file'])] = index['pts_sha256']
    result = dict(question_id=qid, video_id=row['video_id'], sources=sources, old_replay='passed',
        initial_count=len(coarse), source_frame_count=len(pts), plan=plan, refinement_target_upper=targets,
        diagnostic_file=str(dp.relative_to(run)), diagnostic_sha256=sha256(dp), seconds=time.perf_counter()-started)
    durable(dest,result); return result


def precheck(run, workers=8):
    """接口：全量CPU核验；无需GPU或问答，异常保留检查点。"""
    import concurrent.futures
    import shutil
    from videoqa_full.prepare import verify_frozen_rd
    run = Path(run); run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'precheck.lock'):
        verify_frozen_rd(); data=rows(); start=time.perf_counter(); reports=[]
        if (run/'precheck_summary.json').exists():
            saved=read_json(run/'precheck_summary.json')
            for q,h in saved['records'].items():assert sha256(run/'precheck'/(q+'.json'))==h
            return saved
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures=[executor.submit(precheck_question,(str(run),r)) for r in data]
            for future in concurrent.futures.as_completed(futures):
                reports.append(future.result())
                durable(run/'precheck_status.json',dict(completed=len(reports),total=2700,utc=utc(),pid=os.getpid()))
        # 步骤3：按新增PNG保守8MB/帧及轨迹空间预留，不把磁盘缓存称冷读。
        estimate=sum(r['refinement_target_upper'] for r in reports)*8*1024**2+30*1024**3
        free=shutil.disk_usage(run).free
        assert free >= 1.5*estimate, ('Insufficient space',free,estimate)
        summary=dict(status='passed',questions=2700,old_replay_passed=2700,new_model_calls=0,
            short_topk_questions=sum(r['initial_count']<32 for r in reports),
            targets_upper=sum(r['refinement_target_upper'] for r in reports),estimated_new_bytes=estimate,
            free_bytes=free,seconds=time.perf_counter()-start,
            records={r['question_id']:sha256(run/'precheck'/(r['question_id']+'.json')) for r in reports})
        durable(run/'precheck_summary.json',summary)
    return summary


def select_question(run, row, scorer):
    """接口：复用粗资产、重新执行2FPS补查与纯视觉计算，冻结三方法RGB身份。"""
    from PIL import Image
    from videoqa_methods.density_pipeline import decode_regions
    run=Path(run); qid=row['question_id']; dest=run/'selection_checkpoints'/(qid+'.json')
    if dest.exists():
        saved=read_json(dest)
        for rel,h in saved['files'].items(): assert sha256(run/rel)==h
        return saved
    started=time.perf_counter(); old, features, sources=load_old(qid)
    report=read_json(run/'precheck'/(qid+'.json'))
    for p,h in report['sources'].items(): assert sha256(p)==h
    n=old['initial_count']; coarse=old['candidates'][:n]; coarse_features=features[:n].copy()
    plan=build_regions(coarse,old['initial_scores'],old['video'],config()); assert plan==report['plan']
    timing=dict(coarse_asset_read_seconds=time.perf_counter()-started)
    # 步骤1：新增解码仅使用本轮窗口，旧补查特征不复用。
    added,assets,refinement,dt=decode_regions(old['video'],coarse,plan,config(),run/'fine_assets'/qid)
    timing.update(dt); blocks=[coarse_features]; pre=forward=0.; begin=time.perf_counter()
    for offset in range(0,len(assets),16):
        images=[]
        try:
            for a in assets[offset:offset+16]:
                with Image.open(a['path']) as image: images.append(image.convert('RGB'))
            vectors,p,f=scorer.visual_batch(images);blocks.append(vectors);pre+=p;forward+=f
        finally:
            for image in images:image.close()
    timing.update(fine_visual_total_seconds=time.perf_counter()-begin,fine_preprocess_seconds=pre,fine_forward_seconds=forward)
    pool=dict(video=old['video'],candidates=coarse+added,initial_count=n,initial_scores=old['initial_scores'],
        plan=plan,refinement=refinement,new_assets=assets,new_candidate_count=len(added),new_frames_have_itm_scores=False)
    vectors=np.concatenate(blocks); begin=time.perf_counter()
    selected,allocation,trace=two_pass(pool,vectors,config())
    pool.update(selected_indices=selected,selected_frames=[pool['candidates'][i] for i in selected],budget_trace=allocation,selection_trace=trace)
    timing['selection_seconds']=time.perf_counter()-begin
    # 步骤2：两基线不承担补查；三方法源身份按完整PTS索引核验。
    index=read_json(INDEX/'uniform_records'/(row['video_id']+'.json'));pts=np.load(INDEX/index['pts_file'],allow_pickle=False)
    ranked=sorted(range(n),key=lambda i:(-old['initial_scores'][i],coarse[i]['source_pts'],i))[:32]
    chosen={METHODS[0]:native_frames(pts,index['video']),METHODS[1]:sorted([coarse[i] for i in ranked],key=lambda r:r['source_pts']),METHODS[2]:pool['selected_frames']}
    files={};pp=run/'pools'/(qid+'.json');durable(pp,pool);files[str(pp.relative_to(run))]=sha256(pp)
    fp=run/'features'/(qid+'.npy');files[str(fp.relative_to(run))]=save_array(fp,vectors)
    for method,frames in chosen.items():
        for frame in frames: assert int(pts[frame['source_frame_index']])==frame['source_pts']
        selection=dict(video=old['video'],selected_frames=frames);validate_frames(selection)
        images,hashes=FrameProvider(selection).decode()
        for image in images:image.close()
        path=run/'selections'/method/(qid+'.json')
        durable(path,dict(question_id=qid,video_id=row['video_id'],method=method,selection=selection,rgb_sha256=hashes,
            source_hashes=sources,source_video_sha256=index['source_video_sha256'],configuration='M32/K32' if method==METHODS[2] else 'K32'))
        files[str(path.relative_to(run))]=sha256(path)
    # 步骤3：仅完整选择落检查点；本阶段没有真实生成，不占问答预算。
    result=dict(question_id=qid,files=files,sources=sources,timings=timing,seconds=time.perf_counter()-started,
        new_count=len(added),new_itm_calls=0,utc=utc())
    durable(dest,result);return result
