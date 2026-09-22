"""软折中正式流程：复用原扫描及补查，只替换最终竞争函数。"""
import time
import numpy as np
from videoqa_methods.density_pipeline import DensityScorer, scan_original, decode_regions
from videoqa_methods.density_selection import build_regions
from videoqa_audit.density_local_validation import supported_selection as select_joint

def prepare_density(path, question, scorer, cfg, asset_directory):
    """核心接口：从原视频独立执行六环节，返回可复算选择池、特征及不重叠阶段耗时。"""
    from PIL import Image
    # 步骤1～4：粗扫描评分复用视觉前向，冻结种子与最终区域。
    pool, coarse_features, timing = scan_original(path, question, scorer)
    started = time.perf_counter(); plan = build_regions(pool['candidates'],pool['scores'],pool['video'],cfg)
    timing['region_build_seconds'] = time.perf_counter() - started
    # 步骤5：只补查解码，不计算新帧ITM，不改原种子/窗口。
    added, assets, reports, fine_time = decode_regions(pool['video'],pool['candidates'],plan,cfg,asset_directory)
    timing.update(fine_time)
    # 步骤6：只对新帧补齐纯视觉特征，来源不参与最终竞争。
    blocks = [coarse_features]
    timing.update(fine_feature_read_seconds=0.,fine_visual_preprocess_seconds=0.,fine_visual_forward_seconds=0.)
    for offset in range(0,len(assets),16):
        started = time.perf_counter(); images=[]
        for asset in assets[offset:offset+16]:
            with Image.open(asset['path']) as im:
                images.append(im.convert('RGB'))
        timing['fine_feature_read_seconds'] += time.perf_counter()-started
        vectors, pre, forward = scorer.visual_batch(images)
        blocks.append(vectors); timing['fine_visual_preprocess_seconds'] += pre; timing['fine_visual_forward_seconds'] += forward
        for im in images:
            im.close()
    started = time.perf_counter(); features = np.concatenate(blocks,axis=0)
    rows = pool['candidates'] + added
    selected, trace = select_joint(rows,features,plan,pool['video'],cfg)
    timing['selection_compete_seconds'] = time.perf_counter()-started
    result = dict(video=pool['video'],candidates=rows,initial_count=len(pool['candidates']),initial_scores=pool['scores'],
                  blip_question_tokens=pool['blip_question_tokens'],plan=plan,refinement=reports,new_assets=assets,
                  new_candidate_count=len(added),selected_indices=selected,selected_frames=[rows[i] for i in selected],
                  selection_trace=trace,application_cache_hits=0,new_frames_have_itm_scores=False)
    return result, features, timing
